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
def test_cross2d_exceptions(self):
    cfunc = njit(nb_cross2d)
    self.disable_leak_check()
    with self.assertRaises(ValueError) as raises:
        cfunc(np.array((1, 2, 3)), np.array((4, 5, 6)))
    self.assertIn('Incompatible dimensions for 2D cross product', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc(np.arange(6).reshape((2, 3)), np.arange(6)[::-1].reshape((2, 3)))
    self.assertIn('Incompatible dimensions for 2D cross product', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(set([1, 2]), set([4, 5]))
    self.assertIn('Inputs must be array-like.', str(raises.exception))