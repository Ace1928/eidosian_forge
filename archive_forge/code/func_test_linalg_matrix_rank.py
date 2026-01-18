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
@needs_lapack
def test_linalg_matrix_rank(self):
    """
        Test np.linalg.matrix_rank
        """
    cfunc = jit(nopython=True)(matrix_rank_matrix)

    def check(a, **kwargs):
        expected = matrix_rank_matrix(a, **kwargs)
        got = cfunc(a, **kwargs)
        np.testing.assert_allclose(got, expected)
        with self.assertNoNRTLeak():
            cfunc(a, **kwargs)
    sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]
    for size, dtype, order in product(sizes, self.dtypes, 'FC'):
        a = self.specific_sample_matrix(size, dtype, order)
        check(a)
        tol = 1e-13
        for k in range(1, min(size) - 1):
            a = self.specific_sample_matrix(size, dtype, order, rank=k)
            self.assertEqual(cfunc(a), k)
            check(a)
            m, n = a.shape
            a[:, :] = 0.0
            idx = np.nonzero(np.eye(m, n))
            if np.iscomplexobj(a):
                b = 1.0 + np.random.rand(k) + 1j + 1j * np.random.rand(k)
                b[0] = 1e-14 + 1e-14j
            else:
                b = 1.0 + np.random.rand(k)
                b[0] = 1e-14
            a[idx[0][:k], idx[1][:k]] = b.astype(dtype)
            self.assertEqual(cfunc(a, tol), k - 1)
            check(a, tol=tol)
        a[:, :] = 0.0
        self.assertEqual(cfunc(a), 0)
        check(a)
        if np.iscomplexobj(a):
            a[-1, -1] = 1e-14 + 1e-14j
        else:
            a[-1, -1] = 1e-14
        self.assertEqual(cfunc(a, tol), 0)
        check(a, tol=tol)
    for dt in self.dtypes:
        a = np.zeros(5, dtype=dt)
        self.assertEqual(cfunc(a), 0)
        check(a)
        a[0] = 1.0
        self.assertEqual(cfunc(a), 1)
        check(a)
    for sz in [(0, 1), (1, 0), (0, 0)]:
        for tol in [None, 1e-13]:
            self.assert_raise_on_empty(cfunc, (np.empty(sz), tol))
    rn = 'matrix_rank'
    self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32),))
    self.assert_wrong_dimensions_1D(rn, cfunc, (np.ones(12, dtype=np.float64).reshape(2, 2, 3),))
    self.assert_no_nan_or_inf(cfunc, (np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float64),))