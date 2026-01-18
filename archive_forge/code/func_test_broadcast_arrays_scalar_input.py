from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_arrays_scalar_input(self):
    pyfunc = numpy_broadcast_arrays
    cfunc = jit(nopython=True)(pyfunc)
    data = [[[True, False], (1,)], [[1, 2], (1,)], [[(1, 2), 2], (2,)]]
    for inarrays, expected_shape in data:
        outarrays = cfunc(*inarrays)
        got = [a.shape for a in outarrays]
        expected = [expected_shape] * len(inarrays)
        self.assertPreciseEqual(expected, got)