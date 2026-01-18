from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_shapes(self):
    pyfunc = numpy_broadcast_shapes
    cfunc = jit(nopython=True)(pyfunc)
    data = [[()], [(), ()], [(7,)], [(1, 2)], [(1, 1)], [(1, 1), (3, 4)], [(6, 7), (5, 6, 1), (7,), (5, 1, 7)], [(5, 6, 1)], [(1, 3), (3, 1)], [(1, 0), (0, 0)], [(0, 1), (0, 0)], [(1, 0), (0, 1)], [(1, 1), (0, 0)], [(1, 1), (1, 0)], [(1, 1), (0, 1)], [(), (0,)], [(0,), (0, 0)], [(0,), (0, 1)], [(1,), (0, 0)], [(), (0, 0)], [(1, 1), (0,)], [(1,), (0, 1)], [(1,), (1, 0)], [(), (1, 0)], [(), (0, 1)], [(1,), (3,)], [2, (3, 2)]]
    for input_shape in data:
        expected = pyfunc(*input_shape)
        got = cfunc(*input_shape)
        self.assertIsInstance(got, tuple)
        self.assertPreciseEqual(expected, got)