import numpy as np
from numba import cuda, float32, int32, void
from numba.cuda.testing import unittest, CUDATestCase
@cuda.jit(void(float32[:], float32[:], float32[:], int32))
def preCalc(y, yA, yB, numDataPoints):
    i = cuda.grid(1)
    k = i % numDataPoints
    ans = float32(1.001 * float32(i))
    y[i] = ans
    yA[i] = ans * 1.0
    yB[i] = ans / 1.0