from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_shapes_negative_dimension(self):
    pyfunc = numpy_broadcast_shapes
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertRaises(ValueError) as raises:
        cfunc((1, 2), 2, -2)
    self.assertIn('negative dimensions are not allowed', str(raises.exception))