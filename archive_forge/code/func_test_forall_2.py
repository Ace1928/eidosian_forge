import numpy as np
from numba import cuda
import unittest
from numba.cuda.testing import CUDATestCase
def test_forall_2(self):

    @cuda.jit('void(float32, float32[:], float32[:])')
    def bar(a, x, y):
        i = cuda.grid(1)
        if i < x.size:
            y[i] = a * x[i] + y[i]
    x = np.arange(13, dtype=np.float32)
    y = np.arange(13, dtype=np.float32)
    oldy = y.copy()
    a = 1.234
    bar.forall(y.size)(a, x, y)
    np.testing.assert_array_almost_equal(y, a * x + oldy, decimal=3)