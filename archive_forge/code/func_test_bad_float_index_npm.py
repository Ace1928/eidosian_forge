from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_bad_float_index_npm(self):
    with self.assertTypingError() as raises:
        njit((types.Array(types.float64, 2, 'C'),))(bad_float_index)
    self.assertIn('Unsupported array index type float64', str(raises.exception))