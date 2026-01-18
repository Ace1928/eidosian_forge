from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_arrays_tuple_input(self):
    pyfunc = numpy_broadcast_arrays
    cfunc = jit(nopython=True)(pyfunc)
    outarrays = cfunc((123, 456), (789,))
    expected = [(2,), (2,)]
    got = [a.shape for a in outarrays]
    self.assertPreciseEqual(expected, got)