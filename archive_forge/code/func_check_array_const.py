import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase
def check_array_const(self, pyfunc):
    cfunc = njit((types.int32,))(pyfunc)
    for i in [0, 1, 2]:
        np.testing.assert_array_equal(pyfunc(i), cfunc(i))