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
@needs_blas
def test_cov_invalid_ddof(self):
    pyfunc = cov
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    m = np.array([[0, 2], [1, 1], [2, 0]]).T
    for ddof in (np.arange(4), 4j):
        with self.assertTypingError() as raises:
            cfunc(m, ddof=ddof)
        self.assertIn('ddof must be a real numerical scalar type', str(raises.exception))
    for ddof in (np.nan, np.inf):
        with self.assertRaises(ValueError) as raises:
            cfunc(m, ddof=ddof)
        self.assertIn('Cannot convert non-finite ddof to integer', str(raises.exception))
    for ddof in (1.1, -0.7):
        with self.assertRaises(ValueError) as raises:
            cfunc(m, ddof=ddof)
        self.assertIn('ddof must be integral value', str(raises.exception))