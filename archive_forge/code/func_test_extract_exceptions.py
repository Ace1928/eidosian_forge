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
def test_extract_exceptions(self):
    pyfunc = extract
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    a = np.array([])
    cond = np.array([1, 2, 3])
    with self.assertRaises(ValueError) as e:
        cfunc(cond, a)
    self.assertIn('Cannot extract from an empty array', str(e.exception))

    def _check(cond, a):
        msg = 'condition shape inconsistent with arr shape'
        with self.assertRaises(ValueError) as e:
            cfunc(cond, a)
        self.assertIn(msg, str(e.exception))
    a = np.array([[1, 2, 3], [1, 2, 3]])
    cond = [1, 0, 1, 0, 1, 0, 1]
    _check(cond, a)
    a = np.array([1, 2, 3])
    cond = np.array([1, 0, 1, 0, 1])
    _check(cond, a)
    a = np.array(60)
    cond = (0, 1)
    _check(cond, a)
    a = np.arange(4)
    cond = np.array([True, False, False, False, True])
    _check(cond, a)
    a = np.arange(4)
    cond = np.array([True, False, True, False, False, True, False])
    _check(cond, a)