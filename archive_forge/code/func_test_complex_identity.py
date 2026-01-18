import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_complex_identity(self):
    pyfunc = identity
    cfunc = njit(types.complex64(types.complex64))(pyfunc)
    xs = [1j, 1 + 1j, -1 - 1j, 1 + 0j]
    for x in xs:
        self.assertEqual(cfunc(x), x)
    for x in np.complex64(xs):
        self.assertEqual(cfunc(x), x)
    cfunc = njit(types.complex128(types.complex128))(pyfunc)
    xs = [1j, 1 + 1j, -1 - 1j, 1 + 0j]
    for x in xs:
        self.assertEqual(cfunc(x), x)
    for x in np.complex128(xs):
        self.assertEqual(cfunc(x), x)