import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def mk_ctypes_func(sig):

    def ctypes_func(func, sig=int64(int64)):
        assert isinstance(func, pytypes.FunctionType), repr(func)
        cfunc = mk_cfunc_func(sig)(func)
        addr = cfunc._wrapper_address
        if sig == int64(int64):
            f = ctypes.CFUNCTYPE(ctypes.c_int64)(addr)
            f.pyfunc = func
            return f
        raise NotImplementedError(f'ctypes decorator for {func} with signature {sig}')
    return ctypes_func