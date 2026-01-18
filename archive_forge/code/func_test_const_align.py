import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_const_align(self):
    jcuconstAlign = cuda.jit('void(float64[:])')(cuconstAlign)
    A = np.full(3, fill_value=np.nan, dtype=float)
    jcuconstAlign[1, 3](A)
    self.assertTrue(np.all(A == CONST3BYTES + CONST1D[:3]))