from numba import njit
from numba.core import types
import unittest
def test_min3(self):
    pyfunc = domin3
    argtys = (types.int32, types.float32, types.double)
    cfunc = njit(argtys)(pyfunc)
    a = 1
    b = 2
    c = 3
    self.assertEqual(pyfunc(a, b, c), cfunc(a, b, c))