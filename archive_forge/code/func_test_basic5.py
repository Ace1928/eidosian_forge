import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_basic5(self):
    a = 1

    @njit
    def foo1(x):
        return x + 1

    @njit
    def foo2(x):
        return x + 2

    @njit
    def bar1(x):
        return x / 10

    @njit
    def bar2(x):
        return x / 1000
    tup = (foo1, foo2)
    tup_bar = (bar1, bar2)
    int_int_fc = types.FunctionType(types.int64(types.int64))
    flt_flt_fc = types.FunctionType(types.float64(types.float64))

    @njit((types.UniTuple(int_int_fc, 2), types.UniTuple(flt_flt_fc, 2)))
    def bar(fcs, ffs):
        x = 0
        for i in range(2):
            x += fcs[i](a)
        for fn in ffs:
            x += fn(a)
        return x
    got = bar(tup, tup_bar)
    expected = foo1(a) + foo2(a) + bar1(a) + bar2(a)
    self.assertEqual(got, expected)