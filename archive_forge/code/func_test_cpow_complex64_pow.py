import math
import numpy as np
from numba import cuda, float64, int8, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def test_cpow_complex64_pow(self):
    self._test_cpow(np.complex64, vec_pow, rtol=3e-07)