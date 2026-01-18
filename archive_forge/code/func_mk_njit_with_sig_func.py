import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def mk_njit_with_sig_func(sig):

    def njit_with_sig_func(func):
        assert isinstance(func, pytypes.FunctionType), repr(func)
        f = jit(sig, nopython=True)(func)
        f.pyfunc = func
        return f
    return njit_with_sig_func