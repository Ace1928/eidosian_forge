import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def run_array_1d(self, item_type, arg, flags):
    pyfunc = scalar_iter_usecase
    cfunc = jit(item_type(types.Array(item_type, 1, 'A')), **flags)(pyfunc)
    self.assertPreciseEqual(cfunc(arg), pyfunc(arg))