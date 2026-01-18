import numpy as np
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest
def test_0darrayT_to_T(self):

    @njit
    def inner(x):
        return x.dtype.type(x)
    inputs = [(np.bool_, True), (np.float32, 12.3), (np.float64, 12.3), (np.int64, 12), (np.complex64, 2j + 3), (np.complex128, 2j + 3), (np.timedelta64, np.timedelta64(3, 'h')), (np.datetime64, np.datetime64('2016-01-01')), ('<U3', 'ABC')]
    for T, inp in inputs:
        x = np.array(inp, dtype=T)
        self.assertEqual(inner(x), x[()])