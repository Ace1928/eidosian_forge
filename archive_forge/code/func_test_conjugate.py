import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def test_conjugate(self):
    pyfunc = conjugate_usecase
    values = self.basic_values()
    self.run_unary(pyfunc, [types.complex64, types.complex128], values)