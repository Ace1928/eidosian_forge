import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def test_log_base(self):
    values = list(itertools.product(self.more_values(), self.more_values()))
    value_types = [(types.complex128, types.complex128), (types.complex64, types.complex64)]
    self.run_binary(log_base_usecase, value_types, values, ulps=3)