import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_array_ary(self):

    @cuda.jit('double(double[:],int64)', device=True, inline=True)
    def device_function(a, c):
        return a[c]

    @cuda.jit('void(double[:],double[:])')
    def kernel(x, y):
        i = cuda.grid(1)
        y[i] = device_function(x, i)
    x = np.arange(10, dtype=np.double)
    y = np.zeros_like(x)
    kernel[10, 1](x, y)
    self.assertTrue(np.all(x == y))