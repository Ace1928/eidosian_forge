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
def test_myarray_ufunc_unsupported(self):

    @njit
    def foo(buf):
        converted = MyArray(buf.shape, buf.dtype, buf)
        return converted + converted
    buf = np.arange(4, dtype=np.float32)
    with self.assertRaises(TypingError) as raises:
        foo(buf)
    msg = ('No implementation of function', 'add(MyArray(1, float32, C), MyArray(1, float32, C))')
    for m in msg:
        self.assertIn(m, str(raises.exception))