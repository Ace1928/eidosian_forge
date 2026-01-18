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
def test_around_bad_array(self):
    for pyfunc in (np_round_unary, np_around_unary):
        cfunc = jit(nopython=True)(pyfunc)
        msg = '.*The argument "a" must be array-like.*'
        with self.assertRaisesRegex(TypingError, msg):
            cfunc(None)