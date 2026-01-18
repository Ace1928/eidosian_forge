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
def test_around_bad_out(self):
    for py_func in (np_round_array, np_around_array, np_round__array):
        cfunc = jit(nopython=True)(py_func)
        msg = '.*The argument "out" must be an array if it is provided.*'
        with self.assertRaisesRegex(TypingError, msg):
            cfunc(5, 0, out=6)