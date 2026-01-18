import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_compilation_results(self):
    """Turn the existing compilation results of a dispatcher instance to
        first-class functions with precise types.
        """

    @jit(nopython=True)
    def add_template(x, y):
        return x + y
    self.assertEqual(add_template(1, 2), 3)
    self.assertEqual(add_template(1.2, 3.4), 4.6)
    cres1, cres2 = add_template.overloads.values()
    iadd = types.CompileResultWAP(cres1)
    fadd = types.CompileResultWAP(cres2)

    @jit(nopython=True)
    def foo(add, x, y):
        return add(x, y)

    @jit(forceobj=True)
    def foo_obj(add, x, y):
        return add(x, y)
    self.assertEqual(foo(iadd, 3, 4), 7)
    self.assertEqual(foo(fadd, 3.4, 4.5), 7.9)
    self.assertEqual(foo_obj(iadd, 3, 4), 7)
    self.assertEqual(foo_obj(fadd, 3.4, 4.5), 7.9)