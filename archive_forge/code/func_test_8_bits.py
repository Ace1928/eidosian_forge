import numpy as np
from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import TestCase
def test_8_bits(self):
    dtypes = (np.uint8, np.int8)
    inputs = ((1, np.uint8, (1, 1)), (-1, np.int8, (255, -1)))
    self.do_testing(inputs, dtypes)