from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_expand_dims_exceptions(self):
    pyfunc = expand_dims
    cfunc = jit(nopython=True)(pyfunc)
    arr = np.arange(5)
    with self.assertTypingError() as raises:
        cfunc('hello', 3)
    self.assertIn('First argument "a" must be an array', str(raises.exception))
    with self.assertTypingError() as raises:
        cfunc(arr, 'hello')
    self.assertIn('Argument "axis" must be an integer', str(raises.exception))