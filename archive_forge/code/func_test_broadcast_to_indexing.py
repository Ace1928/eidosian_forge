from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_to_indexing(self):
    pyfunc = numpy_broadcast_to_indexing
    cfunc = jit(nopython=True)(pyfunc)
    data = [[np.ones(2), (2, 2), (1,)]]
    for input_array, shape, idx in data:
        expected = pyfunc(input_array, shape, idx)
        got = cfunc(input_array, shape, idx)
        self.assertPreciseEqual(got, expected)