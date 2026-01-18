import numpy as np
import math
from numba import cuda
from numba.types import float32, float64, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def test_frexp_f8(self):
    self.template_test_frexp(np.float64, float64)