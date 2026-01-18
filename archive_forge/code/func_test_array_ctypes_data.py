import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
def test_array_ctypes_data(self):
    pyfunc = array_ctypes_data
    cfunc = njit(pyfunc)
    arr = np.arange(3)
    self.assertEqual(pyfunc(arr), cfunc(arr))