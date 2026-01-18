import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def test_complex2(self):
    pyfunc = docomplex2
    x_types = [types.int32, types.int64, types.float32, types.float64]
    x_values = [1, 1000, 12.2, 23.4]
    y_values = [x - 3 for x in x_values]
    for ty, x, y in zip(x_types, x_values, y_values):
        cfunc = njit((ty, ty))(pyfunc)
        self.assertPreciseEqual(pyfunc(x, y), cfunc(x, y), prec='single' if ty is types.float32 else 'exact')
    pyfunc = complex_calc2
    x = 1.0 + 2 ** (-50)
    cfunc = njit((types.float32, types.float32))(pyfunc)
    self.assertPreciseEqual(cfunc(x, x), 2.0)
    cfunc = njit((types.float64, types.float32))(pyfunc)
    self.assertGreater(cfunc(x, x), 2.0)