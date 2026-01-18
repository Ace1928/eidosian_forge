import numpy as np
from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import TestCase
def test_exceptions32(self):
    for pair in ((np.int32, np.int8), (np.int8, np.int32)):
        self.do_testing_exceptions(pair)