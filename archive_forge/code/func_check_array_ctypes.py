from ctypes import *
import sys
import threading
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.core.typing import ctypes_utils
from numba.tests.support import MemoryLeakMixin, tag, TestCase
from numba.tests.ctypes_usecases import *
import unittest
def check_array_ctypes(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    arr = np.linspace(0, 10, 5)
    expected = arr ** 2.0
    got = cfunc(arr)
    self.assertPreciseEqual(expected, got)
    return cfunc