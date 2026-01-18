import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_issue_5685(self):

    @njit
    def foo1():
        return 1

    @njit
    def foo2(x):
        return x + 1

    @njit
    def foo3(x):
        return x + 2

    @njit
    def bar(fcs):
        r = 0
        for pair in literal_unroll(fcs):
            f1, f2 = pair
            r += f1() + f2(2)
        return r
    self.assertEqual(bar(((foo1, foo2),)), 4)
    self.assertEqual(bar(((foo1, foo2), (foo1, foo3))), 9)