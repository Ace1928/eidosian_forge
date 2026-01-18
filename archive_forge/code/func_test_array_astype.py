from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def test_array_astype(self):

    def run(arr, dtype):
        pyfunc = make_array_astype(dtype)
        return njit(pyfunc)(arr)

    def check(arr, dtype):
        expected = arr.astype(dtype).copy(order='A')
        got = run(arr, dtype)
        self.assertPreciseEqual(got, expected)
    arr = np.arange(24, dtype=np.int8)
    check(arr, np.dtype('int16'))
    check(arr, np.int32)
    check(arr, np.float32)
    check(arr, np.complex128)
    check(arr, 'float32')
    arr = np.arange(24, dtype=np.int8).reshape((3, 8)).T
    check(arr, np.float32)
    arr = np.arange(16, dtype=np.int32)[::2]
    check(arr, np.uint64)
    arr = np.arange(16, dtype=np.int32)
    arr.flags.writeable = False
    check(arr, np.int32)
    dt = np.dtype([('x', np.int8)])
    with self.assertTypingError() as raises:
        check(arr, dt)
    self.assertIn('cannot convert from int32 to Record', str(raises.exception))
    unicode_val = 'float32'
    with self.assertTypingError() as raises:

        @jit(nopython=True)
        def foo(dtype):
            np.array([1]).astype(dtype)
        foo(unicode_val)
    self.assertIn('array.astype if dtype is a string it must be constant', str(raises.exception))