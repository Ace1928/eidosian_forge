import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_basic4(self):
    """
        Test that a dispatcher object can be used as input to another
         function with signature as part of a tuple
        """
    a = 1

    @njit
    def foo1(x):
        return x + 1

    @njit
    def foo2(x):
        return x + 2
    tup = (foo1, foo2)
    int_int_fc = types.FunctionType(types.int64(types.int64))

    @njit(types.int64(types.UniTuple(int_int_fc, 2)))
    def bar(fcs):
        x = 0
        for i in range(2):
            x += fcs[i](a)
        return x
    self.assertEqual(bar(tup), foo1(a) + foo2(a))