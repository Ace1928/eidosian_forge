import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
def check_dot_vv(self, pyfunc, func_name):
    n = 3
    cfunc = jit(nopython=True)(pyfunc)
    for dtype in self.dtypes:
        a = self.sample_vector(n, dtype)
        b = self.sample_vector(n, dtype)
        self.check_func(pyfunc, cfunc, (a, b))
        self.check_func(pyfunc, cfunc, (a[::-1], b[::-1]))
    a = self.sample_vector(n - 1, np.float64)
    b = self.sample_vector(n, np.float64)
    self.assert_mismatching_sizes(cfunc, (a, b))
    a = self.sample_vector(n, np.float32)
    b = self.sample_vector(n, np.float64)
    self.assert_mismatching_dtypes(cfunc, (a, b), func_name=func_name)