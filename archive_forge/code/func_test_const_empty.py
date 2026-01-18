import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_const_empty(self):
    jcuconstEmpty = cuda.jit('void(int64[:])')(cuconstEmpty)
    A = np.full(1, fill_value=-1, dtype=np.int64)
    jcuconstEmpty[1, 1](A)
    self.assertTrue(np.all(A == 0))