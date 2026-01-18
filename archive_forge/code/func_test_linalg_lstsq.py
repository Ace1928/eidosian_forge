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
def test_linalg_lstsq(self):
    """
        Test np.linalg.lstsq
        """
    cfunc = jit(nopython=True)(lstsq_system)

    def check(A, B, **kwargs):
        expected = lstsq_system(A, B, **kwargs)
        got = cfunc(A, B, **kwargs)
        self.assertEqual(len(expected), len(got))
        self.assertEqual(len(got), 4)
        self.assert_contig_sanity(got, 'C')
        use_reconstruction = False
        try:
            self.assertEqual(got[2], expected[2])
            for k in range(len(expected)):
                try:
                    np.testing.assert_array_almost_equal_nulp(got[k], expected[k], nulp=10)
                except AssertionError:
                    use_reconstruction = True
        except AssertionError:
            use_reconstruction = True
        if use_reconstruction:
            x, res, rank, s = got
            out_array_idx = [0, 1, 3]
            try:
                self.assertEqual(rank, expected[2])
                for k in out_array_idx:
                    if isinstance(expected[k], np.ndarray):
                        self.assertEqual(got[k].shape, expected[k].shape)
            except AssertionError:
                self.assertTrue(abs(rank - expected[2]) < 2)
            resolution = np.finfo(A.dtype).resolution
            try:
                rec = np.dot(A, x)
                np.testing.assert_allclose(B, rec, rtol=10 * resolution, atol=10 * resolution)
            except AssertionError:
                for k in out_array_idx:
                    try:
                        np.testing.assert_allclose(expected[k], got[k], rtol=100 * resolution, atol=100 * resolution)
                    except AssertionError:
                        c = np.linalg.cond(A)
                        self.assertGreater(10 * c, 1.0 / resolution)
                    res_expected = np.linalg.norm(B - np.dot(A, expected[0]))
                    res_got = np.linalg.norm(B - np.dot(A, x))
                    np.testing.assert_allclose(res_expected, res_got, rtol=10.0)
        with self.assertNoNRTLeak():
            cfunc(A, B, **kwargs)
    sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]
    cycle_dt = cycle(self.dtypes)
    orders = ['F', 'C']
    cycle_order = cycle(orders)
    specific_cond = 10.0

    def inner_test_loop_fn(A, dt, **kwargs):
        b_sizes = (1, 13)
        for b_size in b_sizes:
            b_order = next(cycle_order)
            B = self.specific_sample_matrix((A.shape[0], b_size), dt, b_order)
            check(A, B, **kwargs)
            b_order = next(cycle_order)
            tmp = B[:, 0].copy(order=b_order)
            check(A, tmp, **kwargs)
    for a_size in sizes:
        dt = next(cycle_dt)
        a_order = next(cycle_order)
        A = self.specific_sample_matrix(a_size, dt, a_order)
        inner_test_loop_fn(A, dt)
        m, n = a_size
        minmn = min(m, n)
        if m != 1 and n != 1:
            r = minmn - 1
            A = self.specific_sample_matrix(a_size, dt, a_order, rank=r)
            inner_test_loop_fn(A, dt)
            A = self.specific_sample_matrix(a_size, dt, a_order, condition=specific_cond)
            rcond = 1.0 / specific_cond
            approx_half_rank_rcond = minmn * rcond
            inner_test_loop_fn(A, dt, rcond=approx_half_rank_rcond)
    empties = [[(0, 1), (1,)], [(1, 0), (1,)], [(1, 1), (0,)], [(1, 1), (1, 0)]]
    for A, b in empties:
        args = (np.empty(A), np.empty(b))
        self.assert_raise_on_empty(cfunc, args)
    ok = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    (cfunc, (ok, ok))
    rn = 'lstsq'
    bad = np.array([[1, 2], [3, 4]], dtype=np.int32)
    self.assert_wrong_dtype(rn, cfunc, (ok, bad))
    self.assert_wrong_dtype(rn, cfunc, (bad, ok))
    bad = np.array([[1, 2], [3, 4]], dtype=np.float32)
    self.assert_homogeneous_dtypes(rn, cfunc, (ok, bad))
    self.assert_homogeneous_dtypes(rn, cfunc, (bad, ok))
    bad = np.array([1, 2], dtype=np.float64)
    self.assert_wrong_dimensions(rn, cfunc, (bad, ok))
    bad = np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float64)
    self.assert_no_nan_or_inf(cfunc, (ok, bad))
    self.assert_no_nan_or_inf(cfunc, (bad, ok))
    oneD = np.array([1.0, 2.0], dtype=np.float64)
    (cfunc, (ok, oneD))
    bad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    self.assert_wrong_dimensions_1D(rn, cfunc, (ok, bad))
    bad1D = np.array([1.0], dtype=np.float64)
    bad2D = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    self.assert_dimensionally_invalid(cfunc, (ok, bad1D))
    self.assert_dimensionally_invalid(cfunc, (ok, bad2D))