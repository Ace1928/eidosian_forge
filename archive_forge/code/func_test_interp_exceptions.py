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
def test_interp_exceptions(self):
    pyfunc = interp
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    x = np.array([1, 2, 3])
    xp = np.array([])
    fp = np.array([])
    with self.assertRaises(ValueError) as e:
        cfunc(x, xp, fp)
    msg = 'array of sample points is empty'
    self.assertIn(msg, str(e.exception))
    x = 1
    xp = np.array([1, 2, 3])
    fp = np.array([1, 2])
    with self.assertRaises(ValueError) as e:
        cfunc(x, xp, fp)
    msg = 'fp and xp are not of the same size.'
    self.assertIn(msg, str(e.exception))
    x = 1
    xp = np.arange(6).reshape(3, 2)
    fp = np.arange(6)
    with self.assertTypingError() as e:
        cfunc(x, xp, fp)
    msg = 'xp must be 1D'
    self.assertIn(msg, str(e.exception))
    x = 1
    xp = np.arange(6)
    fp = np.arange(6).reshape(3, 2)
    with self.assertTypingError() as e:
        cfunc(x, xp, fp)
    msg = 'fp must be 1D'
    self.assertIn(msg, str(e.exception))
    x = 1 + 1j
    xp = np.arange(6)
    fp = np.arange(6)
    with self.assertTypingError() as e:
        cfunc(x, xp, fp)
    complex_dtype_msg = 'Cannot cast array data from complex dtype to float64 dtype'
    self.assertIn(complex_dtype_msg, str(e.exception))
    x = 1
    xp = (np.arange(6) + 1j).astype(np.complex64)
    fp = np.arange(6)
    with self.assertTypingError() as e:
        cfunc(x, xp, fp)
    self.assertIn(complex_dtype_msg, str(e.exception))