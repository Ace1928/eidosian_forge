import builtins
import unittest
from numbers import Number
from functools import wraps
import numpy as np
from llvmlite import ir
import numba
from numba import njit, typeof, objmode
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import (intrinsic, lower_builtin, overload_classmethod,
from numba.np import numpy_support
from numba.tests.support import TestCase, MemoryLeakMixin
def test_myarray_asarray_non_jit(self):

    def foo(buf):
        converted = MyArray(buf.shape, buf.dtype, buf)
        return np.asarray(converted) + buf
    buf = np.arange(4)
    got = foo(buf)
    self.assertIs(type(got), np.ndarray)
    self.assertPreciseEqual(got, buf + buf)