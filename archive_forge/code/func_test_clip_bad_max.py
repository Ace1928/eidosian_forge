from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def test_clip_bad_max(self):
    cfunc = jit(nopython=True)(np_clip)
    msg = '.*The argument "a_max" must be a number.*'
    with self.assertRaisesRegex(TypingError, msg):
        cfunc(1, 1, 'b')