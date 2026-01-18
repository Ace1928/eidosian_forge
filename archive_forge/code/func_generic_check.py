from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def generic_check(pyfunc, a, assume_layout):
    arraytype1 = typeof(a)
    self.assertEqual(arraytype1.layout, assume_layout)
    cfunc = jit((arraytype1,), **flags)(pyfunc)
    expected = pyfunc(a)
    got = cfunc(a)
    np.testing.assert_equal(expected, got)
    py_copied = a.ctypes.data != expected.ctypes.data
    nb_copied = a.ctypes.data != got.ctypes.data
    self.assertEqual(py_copied, assume_layout != 'C')
    self.assertEqual(py_copied, nb_copied)