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
def test_linalg_pinv(self):
    """
        Test np.linalg.pinv
        """
    cfunc = jit(nopython=True)(pinv_matrix)

    def check(a, **kwargs):
        expected = pinv_matrix(a, **kwargs)
        got = cfunc(a, **kwargs)
        self.assert_contig_sanity(got, 'F')
        use_reconstruction = False
        try:
            np.testing.assert_array_almost_equal_nulp(got, expected, nulp=10)
        except AssertionError:
            use_reconstruction = True
        if use_reconstruction:
            self.assertEqual(got.shape, expected.shape)
            rec = np.dot(got, a)
            try:
                self.assert_is_identity_matrix(rec)
            except AssertionError:
                resolution = 5 * np.finfo(a.dtype).resolution
                rec = cfunc(got)
                np.testing.assert_allclose(rec, a, rtol=10 * resolution, atol=100 * resolution)
                if a.shape[0] >= a.shape[1]:
                    lstsq = jit(nopython=True)(lstsq_system)
                    lstsq_pinv = lstsq(a, np.eye(a.shape[0]).astype(a.dtype), **kwargs)[0]
                    np.testing.assert_allclose(got, lstsq_pinv, rtol=10 * resolution, atol=100 * resolution)
                self.assertLess(np.linalg.norm(got - expected), resolution)
        with self.assertNoNRTLeak():
            cfunc(a, **kwargs)
    sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]
    specific_cond = 10.0
    for size, dtype, order in product(sizes, self.dtypes, 'FC'):
        a = self.specific_sample_matrix(size, dtype, order)
        check(a)
        m, n = size
        if m != 1 and n != 1:
            minmn = min(m, n)
            a = self.specific_sample_matrix(size, dtype, order, condition=specific_cond)
            rcond = 1.0 / specific_cond
            approx_half_rank_rcond = minmn * rcond
            check(a, rcond=approx_half_rank_rcond)
    for sz in [(0, 1), (1, 0)]:
        check(np.empty(sz))
    rn = 'pinv'
    self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32),))
    self.assert_wrong_dimensions(rn, cfunc, (np.ones(10, dtype=np.float64),))
    self.assert_no_nan_or_inf(cfunc, (np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float64),))