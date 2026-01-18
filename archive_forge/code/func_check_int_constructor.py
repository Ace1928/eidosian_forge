import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def check_int_constructor(self, pyfunc):
    x_types = [types.boolean, types.int32, types.int64, types.float32, types.float64]
    x_values = [1, 0, 1000, 12.2, 23.4]
    for ty, x in zip(x_types, x_values):
        cfunc = njit((ty,))(pyfunc)
        self.assertPreciseEqual(pyfunc(x), cfunc(x))