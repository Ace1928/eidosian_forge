from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
def test_call_generated(self):
    """
        Test a nested function call to a generated jit function.
        """
    cfunc = jit(nopython=True)(call_generated)
    self.assertPreciseEqual(cfunc(1, 2), (-4, 2))
    self.assertPreciseEqual(cfunc(1j, 2), (1j + 5, 2))