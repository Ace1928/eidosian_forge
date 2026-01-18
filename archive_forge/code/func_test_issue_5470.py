import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_issue_5470(self):

    @njit()
    def foo1():
        return 10

    @njit()
    def foo2():
        return 20
    formulae_foo = (foo1, foo1)

    @njit()
    def bar_scalar(f1, f2):
        return f1() + f2()

    @njit()
    def bar():
        return bar_scalar(*formulae_foo)
    self.assertEqual(bar(), 20)
    formulae_foo = (foo1, foo2)

    @njit()
    def bar():
        return bar_scalar(*formulae_foo)
    self.assertEqual(bar(), 30)