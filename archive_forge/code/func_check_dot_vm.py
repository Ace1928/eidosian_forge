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
def check_dot_vm(self, pyfunc2, pyfunc3, func_name):

    def samples(m, n):
        for order in 'CF':
            a = self.sample_matrix(m, n, np.float64).copy(order=order)
            b = self.sample_vector(n, np.float64)
            yield (a, b)
        for dtype in self.dtypes:
            a = self.sample_matrix(m, n, dtype)
            b = self.sample_vector(n, dtype)
            yield (a, b)
        yield (a[::-1], b[::-1])
    cfunc2 = jit(nopython=True)(pyfunc2)
    if pyfunc3 is not None:
        cfunc3 = jit(nopython=True)(pyfunc3)
    for m, n in [(2, 3), (3, 0), (0, 3)]:
        for a, b in samples(m, n):
            self.check_func(pyfunc2, cfunc2, (a, b))
            self.check_func(pyfunc2, cfunc2, (b, a.T))
        if pyfunc3 is not None:
            for a, b in samples(m, n):
                out = np.empty(m, dtype=a.dtype)
                self.check_func_out(pyfunc3, cfunc3, (a, b), out)
                self.check_func_out(pyfunc3, cfunc3, (b, a.T), out)
    m, n = (2, 3)
    a = self.sample_matrix(m, n - 1, np.float64)
    b = self.sample_vector(n, np.float64)
    self.assert_mismatching_sizes(cfunc2, (a, b))
    self.assert_mismatching_sizes(cfunc2, (b, a.T))
    if pyfunc3 is not None:
        out = np.empty(m, np.float64)
        self.assert_mismatching_sizes(cfunc3, (a, b, out))
        self.assert_mismatching_sizes(cfunc3, (b, a.T, out))
        a = self.sample_matrix(m, m, np.float64)
        b = self.sample_vector(m, np.float64)
        out = np.empty(m - 1, np.float64)
        self.assert_mismatching_sizes(cfunc3, (a, b, out), is_out=True)
        self.assert_mismatching_sizes(cfunc3, (b, a.T, out), is_out=True)
    a = self.sample_matrix(m, n, np.float32)
    b = self.sample_vector(n, np.float64)
    self.assert_mismatching_dtypes(cfunc2, (a, b), func_name)
    if pyfunc3 is not None:
        a = self.sample_matrix(m, n, np.float64)
        b = self.sample_vector(n, np.float64)
        out = np.empty(m, np.float32)
        self.assert_mismatching_dtypes(cfunc3, (a, b, out), func_name)