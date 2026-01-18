from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_to_corner_cases(self):

    @njit
    def _broadcast_to_1():
        return np.broadcast_to('a', (2, 3))
    expected = _broadcast_to_1.py_func()
    got = _broadcast_to_1()
    self.assertPreciseEqual(expected, got)