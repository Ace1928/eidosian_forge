import numpy as np
import math
from numba import cuda
from numba.types import float32, float64, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def template_test_ldexp(self, nptype, nbtype):
    compiled = cuda.jit(void(nbtype[:], nbtype, int32))(simple_ldexp)
    arg = 0.785375
    exp = 2
    aryx = np.zeros(1, dtype=nptype)
    compiled[1, 1](aryx, arg, exp)
    np.testing.assert_array_equal(aryx, nptype(3.1415))
    arg = np.inf
    compiled[1, 1](aryx, arg, exp)
    np.testing.assert_array_equal(aryx, nptype(np.inf))
    arg = np.nan
    compiled[1, 1](aryx, arg, exp)
    np.testing.assert_array_equal(aryx, nptype(np.nan))