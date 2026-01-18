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
def checker_for_linalg_eig(self, name, func, expected_res_len, check_for_domain_change=None):
    """
        Test np.linalg.eig
        """
    n = 10
    cfunc = jit(nopython=True)(func)
    check = self._check_worker(cfunc, name, expected_res_len, check_for_domain_change)
    for dtype, order in product(self.dtypes, 'FC'):
        a = self.sample_matrix(n, dtype, order)
        check(a)
    for ty in [np.float32, np.complex64]:
        check(np.empty((0, 0), dtype=ty))
        self.assert_non_square(cfunc, (np.ones((2, 3), dtype=ty),))
        self.assert_wrong_dtype(name, cfunc, (np.ones((2, 2), dtype=np.int32),))
        self.assert_wrong_dimensions(name, cfunc, (np.ones(10, dtype=ty),))
        self.assert_no_nan_or_inf(cfunc, (np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=ty),))
    if check_for_domain_change:
        A = np.array([[1, -2], [2, 1]])
        check(A.astype(np.complex128))
        l, _ = func(A)
        self.assertTrue(np.any(l.imag))
        for ty in [np.float32, np.float64]:
            self.assert_no_domain_change(name, cfunc, (A.astype(ty),))