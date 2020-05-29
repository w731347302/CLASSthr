#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

constexpr auto cell_size = 16;
constexpr auto angle_size = 8;

using namespace std;
using namespace cv;


bool cal_gxy(Mat src, Mat &angle, Mat &mag)
{
	Mat gx, gy;
	Sobel(src, gx, CV_32F, 1, 0, 1);
	Sobel(src, gy, CV_32F, 0, 1, 1);
	cartToPolar(gx, gy, mag, angle, true);

	float s = 360 / angle_size;
	for (int i = 0; i < angle.rows; i++)
	{
		for (int j = 0; j < angle.cols; j++)
		{
			float x = angle.at<float>(i, j) / s;
			angle.at<float>(i, j) = x;
		}
	}
	return true;
}

bool divid_img(Mat src, vector<Mat> &cells)
{
	int nX = src.cols / cell_size;
	int nY = src.rows / cell_size;

	for (int i = 0; i < nY; i++)
	{
		for (int j = 0; j < nX; j++)
		{
			Rect rect(j*cell_size, i*cell_size, cell_size, cell_size);
			cells.push_back(Mat(src, rect));
		}
	}

	return  true;
}

bool divid_img(Mat src, vector<Mat> &cells, int t_row, int t_col,vector<Rect> &rects)
{
	int nX = src.cols / t_col;
	int nY = src.rows / t_row;
	//int nX = src.cols;
	//int nY = src.rows;
	//int i = 0;
	  //固定区块划分，快，但是不准确
	for(int i =0;i<nY;i++)
	{
		for(int j = 0;j<nX;j++)
		{
			Rect rect(j*t_col, i*t_row, t_col, t_row);
			rects.push_back(rect);
			cells.push_back(Mat(src, rect));
		}
	}
	
	//逐像素划分，很慢,比较准
	/*
	while (i < nY && i+t_row<nY)
	{
		for (int j = 0;j < nX && j+t_col<nX;j++)
		{
			Rect rect(j, i, t_col, t_row);
			rects.push_back(rect);
			cells.push_back(Mat(src, rect));
		}
		i++;
	}
	*/

	return  true;
}

bool creat_hist(Mat src, vector<float> &hist)
{
	vector<Mat> cells;
	vector<Mat> mag_cells;
	Mat angle, mag;

	cal_gxy(src, angle, mag);
	divid_img(angle, cells);
	divid_img(mag, mag_cells);

	int cells_size = cells.size();
	vector<vector<float>> hist_part(cells_size, vector<float>(angle_size, 0));
	for (int i = 0; i < cells_size; i++)
	{
		for (int m = 0; m < cells[i].rows; m++)
		{
			for (int n = 0; n < cells[i].cols; n++)
			{
				hist_part[i][cells[i].at<float>(m, n)] += mag_cells[i].at<float>(m, n);
			}
		}
	}


	for (int j = 0; j < hist_part.size(); j++)
	{
		for (int k = 0; k < hist_part[0].size(); k++)
		{
			hist[k] += hist_part[j][k];
		}
	}

	return true;
}


float calculation(vector<float> hist1, vector<float> hist2)
{
	int a = hist1.size();
	int b = hist2.size();
	int len = min(a, b);
	float res = 0;
	for (int i = 0; i < len; i++)
	{
		res += (hist1[i] - hist2[i])*(hist1[i] - hist2[i]);
	}
	res = sqrt(res);
	return res;
}

bool match()
{

	VideoCapture cap(0);

	Mat frame;
	Mat temp;
	Mat res;
	Mat refMat;
	Mat dispMat;

	int cnt = 0;
	while (1) {

		cap >> frame;
		if (frame.empty())break;

		if (cnt == 0) {
			Rect2d r;
			r = selectROI(frame, true);
			temp = frame(r);
			temp.copyTo(refMat);
			destroyAllWindows();
		}
		matchTemplate(frame, refMat, res, 0);
		normalize(res, res, 0, 1, NORM_MINMAX, -1, Mat());

		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point matchLoc;

		minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		matchLoc = minLoc;
		frame.copyTo(dispMat);
		rectangle(dispMat, matchLoc, Point(matchLoc.x + refMat.cols, matchLoc.y + refMat.rows), Scalar::all(0), 2, 8, 0);


		cnt++;
		imshow("template", refMat);
		imshow("dispMat", dispMat);
		waitKey(30);

	}

	return true;
}

int main()
{
	Mat img = imread("img.png");
	Mat train_src = imread("template.png",0);
	Mat img1;
	cvtColor(img, img1, COLOR_BGR2GRAY);
	

	float min = INT_MAX;
	int t_row = train_src.rows;
	int t_col = train_src.cols;
	vector<Mat> imgs;
	vector<Rect> rects;
	Mat resimg;
	int flag;
	divid_img(img1, imgs, t_row, t_col,rects);
	vector<float> train_hist(angle_size);
	creat_hist(train_src, train_hist);

	for (int i=0;i<imgs.size();i++)
	{
		vector<float> hist(angle_size);
		creat_hist(imgs[i], hist);
		float res = calculation(hist, train_hist);
		if (res < min)
		{
			min = res;
			resimg = imgs[i];
			flag = i;
		}
	}
	Rect fin_rect = rects[flag];
	rectangle(img, fin_rect, Scalar(255, 0, 0));
	imshow("res", img);
	waitKey(0);
	//match();
	return 0;
}