from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_array_transpose_axes(self):
    pyfuncs_to_use = [numpy_transpose_array_axes_kwarg, numpy_transpose_array_axes_kwarg_copy, array_transpose_axes, array_transpose_axes_copy]

    @from_generic(pyfuncs_to_use)
    def check(pyfunc, arr, axes):
        expected = pyfunc.py_func(arr, axes)
        got = pyfunc(arr, axes)
        self.assertPreciseEqual(got, expected)
        self.assertEqual(got.flags.f_contiguous, expected.flags.f_contiguous)
        self.assertEqual(got.flags.c_contiguous, expected.flags.c_contiguous)

    @from_generic(pyfuncs_to_use)
    def check_err_axis_repeated(pyfunc, arr, axes):
        with self.assertRaises(ValueError) as raises:
            pyfunc(arr, axes)
        self.assertEqual(str(raises.exception), 'repeated axis in transpose')

    @from_generic(pyfuncs_to_use)
    def check_err_axis_oob(pyfunc, arr, axes):
        with self.assertRaises(ValueError) as raises:
            pyfunc(arr, axes)
        self.assertEqual(str(raises.exception), 'axis is out of bounds for array of given dimension')

    @from_generic(pyfuncs_to_use)
    def check_err_invalid_args(pyfunc, arr, axes):
        with self.assertRaises((TypeError, TypingError)):
            pyfunc(arr, axes)
    arrs = [np.arange(24), np.arange(24).reshape(4, 6), np.arange(24).reshape(2, 3, 4), np.arange(24).reshape(1, 2, 3, 4), np.arange(64).reshape(8, 4, 2)[::3, ::2, :]]
    for i in range(len(arrs)):
        check(arrs[i], None)
        for axes in permutations(tuple(range(arrs[i].ndim))):
            ndim = len(axes)
            neg_axes = tuple([x - ndim for x in axes])
            check(arrs[i], axes)
            check(arrs[i], neg_axes)

    @from_generic([transpose_issue_4708])
    def check_issue_4708(pyfunc, m, n):
        expected = pyfunc.py_func(m, n)
        got = pyfunc(m, n)
        np.testing.assert_equal(got, expected)
    check_issue_4708(3, 2)
    check_issue_4708(2, 3)
    check_issue_4708(5, 4)
    self.disable_leak_check()
    check_err_invalid_args(arrs[1], 'foo')
    check_err_invalid_args(arrs[1], ('foo',))
    check_err_invalid_args(arrs[1], 5.3)
    check_err_invalid_args(arrs[2], (1.2, 5))
    check_err_axis_repeated(arrs[1], (0, 0))
    check_err_axis_repeated(arrs[2], (2, 0, 0))
    check_err_axis_repeated(arrs[3], (3, 2, 1, 1))
    check_err_axis_oob(arrs[0], (1,))
    check_err_axis_oob(arrs[0], (-2,))
    check_err_axis_oob(arrs[1], (0, 2))
    check_err_axis_oob(arrs[1], (-3, 2))
    check_err_axis_oob(arrs[1], (0, -3))
    check_err_axis_oob(arrs[2], (3, 1, 2))
    check_err_axis_oob(arrs[2], (-4, 1, 2))
    check_err_axis_oob(arrs[3], (3, 1, 2, 5))
    check_err_axis_oob(arrs[3], (3, 1, 2, -5))
    with self.assertRaises(TypingError) as e:
        jit(nopython=True)(numpy_transpose_array)((np.array([0, 1]),))
    self.assertIn('np.transpose does not accept tuples', str(e.exception))