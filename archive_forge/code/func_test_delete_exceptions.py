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
def test_delete_exceptions(self):
    pyfunc = delete
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc([1, 2], 3.14)
    self.assertIn('obj should be of Integer dtype', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.arange(10), [3.5, 5.6, 6.2])
    self.assertIn('obj should be of Integer dtype', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(2, 3)
    self.assertIn('arr must be either an Array or a Sequence', str(raises.exception))
    with self.assertRaises(IndexError) as raises:
        cfunc([1, 2], 3)
    self.assertIn('obj must be less than the len(arr)', str(raises.exception))
    self.disable_leak_check()