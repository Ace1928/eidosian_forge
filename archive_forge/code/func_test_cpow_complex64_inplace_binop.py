import math
import numpy as np
from numba import cuda, float64, int8, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def test_cpow_complex64_inplace_binop(self):
    self._test_cpow_inplace_binop(np.complex64, rtol=3e-07)