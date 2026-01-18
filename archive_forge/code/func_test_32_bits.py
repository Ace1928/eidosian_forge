import numpy as np
from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import TestCase
def test_32_bits(self):
    dtypes = (np.uint32, np.int32, np.float32)
    inputs = ((1, np.uint32, (1, 1, 1.401298464324817e-45)), (-1, np.int32, (4294967295, -1, np.nan)), (1.0, np.float32, (1065353216, 1065353216, 1.0)))
    self.do_testing(inputs, dtypes)