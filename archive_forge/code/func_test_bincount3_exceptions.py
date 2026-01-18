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
def test_bincount3_exceptions(self):
    pyfunc = bincount3
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertRaises(ValueError) as raises:
        cfunc([2, -1], [0, 0])
    self.assertIn('first argument must be non-negative', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc([17, 38], None, -1)
    self.assertIn("'minlength' must not be negative", str(raises.exception))