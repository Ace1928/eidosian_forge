from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_arrays_same_input_shapes(self):
    pyfunc = numpy_broadcast_arrays
    cfunc = jit(nopython=True)(pyfunc)
    data = [(1,), (3,), (0, 1), (0, 3), (1, 0), (3, 0), (1, 3), (3, 1), (3, 3)]
    for shape in data:
        input_shapes = [shape]
        self.broadcast_arrays_assert_correct_shape(input_shapes, shape)
        input_shapes2 = [shape, shape]
        self.broadcast_arrays_assert_correct_shape(input_shapes2, shape)
        input_shapes3 = [shape, shape, shape]
        self.broadcast_arrays_assert_correct_shape(input_shapes3, shape)