from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def check_as_strided(self, pyfunc):
    cfunc = njit(pyfunc)

    def check(arr):
        expected = pyfunc(arr)
        got = cfunc(arr)
        self.assertPreciseEqual(got, expected)
    arr = np.arange(24)
    check(arr)
    check(arr.reshape((6, 4)))
    check(arr.reshape((4, 1, 6)))