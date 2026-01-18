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
def check_dot_mm(self, pyfunc2, pyfunc3, func_name):

    def samples(m, n, k):
        for order_a, order_b in product('CF', 'CF'):
            a = self.sample_matrix(m, k, np.float64).copy(order=order_a)
            b = self.sample_matrix(k, n, np.float64).copy(order=order_b)
            yield (a, b)
        for dtype in self.dtypes:
            a = self.sample_matrix(m, k, dtype)
            b = self.sample_matrix(k, n, dtype)
            yield (a, b)
        yield (a[::-1], b[::-1])
    cfunc2 = jit(nopython=True)(pyfunc2)
    if pyfunc3 is not None:
        cfunc3 = jit(nopython=True)(pyfunc3)
    for m, n, k in [(2, 3, 4), (1, 3, 4), (1, 1, 4), (0, 3, 2), (3, 0, 2), (0, 0, 3), (3, 2, 0)]:
        for a, b in samples(m, n, k):
            self.check_func(pyfunc2, cfunc2, (a, b))
            self.check_func(pyfunc2, cfunc2, (b.T, a.T))
        if pyfunc3 is not None:
            for a, b in samples(m, n, k):
                out = np.empty((m, n), dtype=a.dtype)
                self.check_func_out(pyfunc3, cfunc3, (a, b), out)
                out = np.empty((n, m), dtype=a.dtype)
                self.check_func_out(pyfunc3, cfunc3, (b.T, a.T), out)
    m, n, k = (2, 3, 4)
    a = self.sample_matrix(m, k - 1, np.float64)
    b = self.sample_matrix(k, n, np.float64)
    self.assert_mismatching_sizes(cfunc2, (a, b))
    if pyfunc3 is not None:
        out = np.empty((m, n), np.float64)
        self.assert_mismatching_sizes(cfunc3, (a, b, out))
        a = self.sample_matrix(m, k, np.float64)
        b = self.sample_matrix(k, n, np.float64)
        out = np.empty((m, n - 1), np.float64)
        self.assert_mismatching_sizes(cfunc3, (a, b, out), is_out=True)
    a = self.sample_matrix(m, k, np.float32)
    b = self.sample_matrix(k, n, np.float64)
    self.assert_mismatching_dtypes(cfunc2, (a, b), func_name)
    if pyfunc3 is not None:
        a = self.sample_matrix(m, k, np.float64)
        b = self.sample_matrix(k, n, np.float64)
        out = np.empty((m, n), np.float32)
        self.assert_mismatching_dtypes(cfunc3, (a, b, out), func_name)