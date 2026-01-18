import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_array_iter_yielded_order(self):

    @jit(nopython=True)
    def foo(arr):
        t = []
        for y1 in arr:
            for y2 in y1:
                t.append(y2.ravel())
        return t
    arr = np.arange(24).reshape((2, 3, 4), order='F')
    expected = foo.py_func(arr)
    got = foo(arr)
    self.assertPreciseEqual(expected, got)
    arr = np.arange(64).reshape((4, 8, 2), order='F')[::2, :, :]
    expected = foo.py_func(arr)
    got = foo(arr)
    self.assertPreciseEqual(expected, got)
    arr = np.arange(64).reshape((4, 8, 2), order='F')[:, ::2, :]
    expected = foo.py_func(arr)
    got = foo(arr)
    self.assertPreciseEqual(expected, got)
    arr = np.arange(64).reshape((4, 8, 2), order='F')[:, :, ::2]
    expected = foo.py_func(arr)
    got = foo(arr)
    self.assertPreciseEqual(expected, got)

    @jit(nopython=True)
    def flag_check(arr):
        out = []
        for sub in arr:
            out.append((sub, sub.flags.c_contiguous, sub.flags.f_contiguous))
        return out
    arr = np.arange(10).reshape((2, 5), order='F')
    expected = flag_check.py_func(arr)
    got = flag_check(arr)
    self.assertEqual(len(expected), len(got))
    ex_arr, e_flag_c, e_flag_f = expected[0]
    go_arr, g_flag_c, g_flag_f = got[0]
    np.testing.assert_allclose(ex_arr, go_arr)
    self.assertEqual(e_flag_c, g_flag_c)
    self.assertEqual(e_flag_f, g_flag_f)