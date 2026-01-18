from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_bad_index_npm(self):
    with self.assertTypingError() as raises:
        arraytype1 = from_dtype(np.dtype([('x', np.int32), ('y', np.int32)]))
        arraytype2 = types.Array(types.int32, 2, 'C')
        njit((arraytype1, arraytype2))(bad_index)
    self.assertIn('Unsupported array index type', str(raises.exception))