import math
import numpy as np
from numba import cuda, float64, int8, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def cu_mat_power(A, power, power_A):
    y, x = cuda.grid(2)
    m, n = power_A.shape
    if x >= n or y >= m:
        return
    power_A[y, x] = math.pow(A[y, x], int32(power))