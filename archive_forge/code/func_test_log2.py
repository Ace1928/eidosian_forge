import numpy as np
from numba import float32, jit, njit
from numba.np.ufunc import Vectorize
from numba.core.errors import TypingError
from numba.tests.support import TestCase
import unittest
def test_log2(self):

    @njit
    def foo(x):
        return np.log2(x)
    for ty in (np.int8, np.uint16):
        x = ty(2)
        expected = foo.py_func(x)
        got = foo(x)
        self.assertPreciseEqual(expected, got)