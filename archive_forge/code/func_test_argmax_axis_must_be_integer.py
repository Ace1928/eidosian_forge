from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_argmax_axis_must_be_integer(self):
    arr = np.arange(6)

    @jit(nopython=True)
    def jitargmax(arr, axis):
        return np.argmax(arr, axis)
    with self.assertTypingError() as e:
        jitargmax(arr, 'foo')
    self.assertIn('axis must be an integer', str(e.exception))