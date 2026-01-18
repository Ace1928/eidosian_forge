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
def test_logspace3_exception(self):
    cfunc = jit(nopython=True)(logspace3)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc('abc', 5)
    self.assertIn('The first argument "start" must be a number', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(5, 'abc')
    self.assertIn('The second argument "stop" must be a number', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(0, 5, 'abc')
    self.assertIn('The third argument "num" must be an integer', str(raises.exception))