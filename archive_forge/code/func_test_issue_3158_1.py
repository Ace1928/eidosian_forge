import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_issue_3158_1(self):
    item = np.dtype([('some_field', np.int32)])
    items = np.dtype([('items', item, 3)])

    @njit
    def fn(x):
        return x[0]
    arr = np.asarray([([(0,), (1,), (2,)],), ([(3,), (4,), (5,)],)], dtype=items)
    expected = fn.py_func(arr)
    actual = fn(arr)
    self.assertEqual(expected, actual)