from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_to_array_attrs(self):

    @njit
    def foo(arr):
        ret = np.broadcast_to(arr, (2, 3))
        return (ret, ret.size, ret.shape, ret.strides)
    arr = np.arange(3)
    expected = foo.py_func(arr)
    got = foo(arr)
    self.assertPreciseEqual(expected, got)