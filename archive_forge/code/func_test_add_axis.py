from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_add_axis(self):

    @njit
    def np_new_axis_getitem(a, idx):
        return a[idx]

    @njit
    def np_new_axis_setitem(a, idx, item):
        a[idx] = item
        return a
    a = np.arange(4 * 5 * 6 * 7).reshape((4, 5, 6, 7))
    idx_cases = [(slice(None), np.newaxis), (np.newaxis, slice(None)), (slice(1), np.newaxis, 1), (np.newaxis, 2, slice(None)), (slice(1), Ellipsis, np.newaxis, 1), (1, np.newaxis, Ellipsis), (np.newaxis, slice(1), np.newaxis, 1), (1, Ellipsis, None, np.newaxis), (np.newaxis, slice(1), Ellipsis, np.newaxis, 1), (1, np.newaxis, np.newaxis, Ellipsis), (np.newaxis, 1, np.newaxis, Ellipsis), (slice(3), 1, np.newaxis, None), (np.newaxis, 1, Ellipsis, None)]
    pyfunc_getitem = np_new_axis_getitem.py_func
    cfunc_getitem = np_new_axis_getitem
    pyfunc_setitem = np_new_axis_setitem.py_func
    cfunc_setitem = np_new_axis_setitem
    for idx in idx_cases:
        expected = pyfunc_getitem(a, idx)
        got = cfunc_getitem(a, idx)
        np.testing.assert_equal(expected, got)
        a_empty = np.zeros_like(a)
        item = a[idx]
        expected = pyfunc_setitem(a_empty.copy(), idx, item)
        got = cfunc_setitem(a_empty.copy(), idx, item)
        np.testing.assert_equal(expected, got)