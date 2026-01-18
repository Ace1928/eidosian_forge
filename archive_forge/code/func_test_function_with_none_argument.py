import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_function_with_none_argument(self):

    @cfunc(int64(types.none))
    def a(i):
        return 1

    @jit(nopython=True)
    def foo(f):
        return f(None)
    self.assertEqual(foo(a), 1)