import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_cross(self):
    pyfunc = np_cross
    cfunc = jit(nopython=True)(pyfunc)
    pairs = [(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[4, 5, 6], [1, 2, 3]])), (np.array([[1, 2, 3], [4, 5, 6]]), ((4, 5), (1, 2))), (np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.float64)), ((1, 2, 3), (4, 5, 6)), (np.array([1, 2]), np.array([4, 5, 6])), (np.array([1, 2, 3]), np.array([[4, 5, 6], [1, 2, 3]])), (np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2, 3])), (np.arange(36).reshape(6, 2, 3), np.arange(4).reshape(2, 2))]
    for x, y in pairs:
        expected = pyfunc(x, y)
        got = cfunc(x, y)
        self.assertPreciseEqual(expected, got)