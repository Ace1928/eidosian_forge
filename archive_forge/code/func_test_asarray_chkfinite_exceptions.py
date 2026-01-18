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
def test_asarray_chkfinite_exceptions(self):
    cfunc = jit(nopython=True)(np_asarray_chkfinite)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as e:
        cfunc(2)
    msg = 'The argument to np.asarray_chkfinite must be array-like'
    self.assertIn(msg, str(e.exception))
    with self.assertRaises(ValueError) as e:
        cfunc(np.array([2, 4, np.nan, 5]))
    self.assertIn('array must not contain infs or NaNs', str(e.exception))
    with self.assertRaises(ValueError) as e:
        cfunc(np.array([1, 2, np.inf, 4]))
    self.assertIn('array must not contain infs or NaNs', str(e.exception))
    with self.assertRaises(TypingError) as e:
        cfunc(np.array([1, 2, 3, 4]), 'float32')
    self.assertIn('dtype must be a valid Numpy dtype', str(e.exception))