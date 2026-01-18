import numpy as np
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest
def test_array_to_array(self):
    """Make sure this compiles.

        Cast C to A array
        """

    @njit('f8(f8[:])')
    def inner(x):
        return x[0]
    inner.disable_compile()

    @njit('f8(f8[::1])')
    def driver(x):
        return inner(x)
    x = np.array([1234], dtype=np.float64)
    self.assertEqual(driver(x), x[0])
    self.assertEqual(len(inner.overloads), 1)