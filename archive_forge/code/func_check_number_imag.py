import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
def check_number_imag(self, dtype):
    pyfunc = array_imag
    cfunc = njit(pyfunc)
    size = 10
    arr = np.arange(size, dtype=dtype)
    self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
    arr = arr.reshape(2, 5)
    self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
    self.assertEqual(cfunc(arr).tolist(), np.zeros_like(arr).tolist())
    imag = cfunc(arr)
    with self.assertRaises(ValueError) as raises:
        imag[0] = 1
    self.assertEqual('assignment destination is read-only', str(raises.exception))