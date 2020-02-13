#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>

#include <cuda_runtime.h>
#include <curand_kernel.h>
// to remove intellisense highlighting
#include <device_launch_parameters.h>

const int ntpb = 512;
using namespace std;
using namespace std::chrono;

void calculatePI(int n, float* h_a) {
	float x, y;
	int hit;
	srand(time(NULL));
	for (int j = 0; j < n; j++) {
		hit = 0;
		x = 0; 
		y = 0;
		for (int i = 0; i < n; i++) {
			x = float(rand()) / float(RAND_MAX);
			y = float(rand()) / float(RAND_MAX);
			if (y <= sqrt(1 - (x * x))) {
				hit += 1;
			}
		}

		h_a[j] = 4 * float(hit) / float(n);

	}
}

__global__ void setRng(curandState *rng) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(123456, idx, 0, &rng[idx]);
}


__global__ void calPI(float* d_a, int n, curandState *rng) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int counter = 0;
	while (counter < n) {
		float x = curand_uniform(&rng[idx]);
		float y = curand_uniform(&rng[idx]);

		if (y <= sqrt(1 - (x * x))) {
			d_a[idx]++;
		}
		counter++;
	}
	d_a[idx] = 4.0 * (float(d_a[idx])) / float(n);
}



void reportTime(const char* msg, steady_clock::duration span) {
	auto ms = duration_cast<milliseconds>(span);
	std::cout << msg << " took - " <<
		ms.count() << " millisecs" << std::endl;
}
int main(int argc, char* argv[]) {

	if (argc != 2) {
		std::cerr << argv[0] << ": invalid number of arguments\n";
		std::cerr << "Usage: " << argv[0] << "  size_of_matrices\n";
		return 1;
	}
	int n = std::atoi(argv[1]); // scale
	int nblks = (n + ntpb - 1) / ntpb;
	cout << "scale: " << n << endl << endl;
	steady_clock::time_point ts, te;

	float* cpu_a;
	cpu_a = new float[n];

	ts = steady_clock::now();
	calculatePI(n, cpu_a);
	te = steady_clock::now();
	reportTime("CPU", te - ts);




	ofstream h_file;
	h_file.open("h_result.txt");
	float cpuSum = 0.0f;
	for (int i = 0; i < n; i++) {
		cpuSum += cpu_a[i];
		h_file << "Host: " << cpu_a[i] << endl;
	}
	cpuSum = cpuSum / (float)n;
	cout << "CPU Result: " << cpuSum << endl;
	h_file.close();

	cout << endl;
	////////////////////////////////////////

	curandState *d_rng;
	float* d_a;
	float* h_a;
	h_a = new float[n];

	cudaMalloc((void**)&d_a, n * sizeof(float));
	cudaMalloc((void**)&d_rng, n * sizeof(curandState));

	ts = steady_clock::now();

	setRng << < nblks, ntpb >> > (d_rng);
	cudaDeviceSynchronize();	// synchronize [new added]
	calPI << <nblks, ntpb >> > (d_a, n, d_rng);
	cudaDeviceSynchronize();

	te = steady_clock::now();
	reportTime("GPU", te - ts);

	cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);


	ofstream d_file;
	d_file.open("d_result.txt");
	float gpuSum = 0.0f;
	for (int i = 0; i < n; i++) {
		gpuSum += h_a[i];
		d_file << "Device: " << h_a[i] << endl;
	}
	gpuSum = gpuSum / (float)n;
	cout << "GPU Result: " << gpuSum << endl;
	d_file.close();


	delete[] cpu_a;
	delete[] h_a;
	cudaFree(d_a);
	cudaFree(d_rng);

}


