import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_issue_3405_using_njit(self):

    @jit(nopython=True)
    def a():
        return 2

    @jit(nopython=True)
    def b():
        return 3

    def g(arg):
        if not arg:
            f = b
        else:
            f = a
        return f()
    self.assertEqual(jit(nopython=True)(g)(True), 2)
    self.assertEqual(jit(nopython=True)(g)(False), 3)