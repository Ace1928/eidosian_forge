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
def test_array_equal(self):

    def arrays():
        yield (np.array([]), np.array([]))
        yield (np.array([1, 2]), np.array([1, 2]))
        yield (np.array([]), np.array([1]))
        x = np.arange(10).reshape(5, 2)
        x[1][1] = 30
        yield (np.arange(10).reshape(5, 2), x)
        yield (x, x)
        yield ((1, 2, 3), (1, 2, 3))
        yield (2, 2)
        yield (3, 2)
        yield (True, True)
        yield (True, False)
        yield (True, 2)
        yield (True, 1)
        yield (False, 0)
    pyfunc = array_equal
    cfunc = jit(nopython=True)(pyfunc)
    for arr, obj in arrays():
        expected = pyfunc(arr, obj)
        got = cfunc(arr, obj)
        self.assertPreciseEqual(expected, got)