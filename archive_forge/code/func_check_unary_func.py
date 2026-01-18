import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def check_unary_func(self, pyfunc, ulps=1, values=None, returns_float=False, ignore_sign_on_zero=False):
    if returns_float:

        def sig(tp):
            return tp.underlying_float(tp)
    else:

        def sig(tp):
            return tp(tp)
    self.run_unary(pyfunc, [sig(types.complex128)], values or self.more_values(), ulps=ulps, ignore_sign_on_zero=ignore_sign_on_zero)
    self.run_unary(pyfunc, [sig(types.complex64)], values or self.basic_values(), ulps=ulps, ignore_sign_on_zero=ignore_sign_on_zero)