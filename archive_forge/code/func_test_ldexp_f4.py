import numpy as np
import math
from numba import cuda
from numba.types import float32, float64, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def test_ldexp_f4(self):
    self.template_test_ldexp(np.float32, float32)