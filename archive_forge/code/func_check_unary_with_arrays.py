import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
def check_unary_with_arrays(self, pyfunc):
    self.check_unary(pyfunc, self.a)
    self.check_unary(pyfunc, self.a.T)
    self.check_unary(pyfunc, self.a[::2])
    arr = np.array([42]).reshape(())
    self.check_unary(pyfunc, arr)
    arr = np.zeros(0)
    self.check_unary(pyfunc, arr)
    self.check_unary(pyfunc, arr.reshape((1, 0, 2)))