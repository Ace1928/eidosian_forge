import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
def check_number_real(self, dtype):
    pyfunc = array_real
    cfunc = njit(pyfunc)
    size = 10
    arr = np.arange(size, dtype=dtype)
    self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
    arr = arr.reshape(2, 5)
    self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
    self.assertEqual(arr.data, pyfunc(arr).data)
    self.assertEqual(arr.data, cfunc(arr).data)
    real = cfunc(arr)
    self.assertNotEqual(arr[0, 0], 5)
    real[0, 0] = 5
    self.assertEqual(arr[0, 0], 5)