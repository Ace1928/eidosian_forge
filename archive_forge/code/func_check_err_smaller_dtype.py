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
def check_err_smaller_dtype(arr, dtype):
    msg = 'When changing to a smaller dtype, its size must be a divisor of the size of original dtype'
    with self.assertRaises(ValueError) as raises:
        make_array_view(dtype)(arr)
    self.assertEqual(str(raises.exception), msg)
    with self.assertRaises(ValueError) as raises:
        run(arr, dtype)
    self.assertEqual(str(raises.exception), msg)