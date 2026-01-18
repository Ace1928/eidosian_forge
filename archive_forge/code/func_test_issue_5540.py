import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_issue_5540(self):

    @njit(types.int64(types.int64))
    def foo(x):
        return x + 1

    @njit
    def bar_bad(foos):
        f = foos[0]
        return f(x=1)

    @njit
    def bar_good(foos):
        f = foos[0]
        return f(1)
    self.assertEqual(bar_good((foo,)), 2)
    with self.assertRaises((errors.UnsupportedError, errors.TypingError)) as cm:
        bar_bad((foo,))
    self.assertRegex(str(cm.exception), '.*first-class function call cannot use keyword arguments')