from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_mutability_after_ravel(self):
    self.disable_leak_check()
    a_c = np.arange(9).reshape((3, 3)).copy()
    a_f = a_c.copy(order='F')
    a_c.flags.writeable = False
    a_f.flags.writeable = False

    def try_ravel_w_copy(a):
        result = a.ravel()
        return result
    pyfunc = try_ravel_w_copy
    cfunc = jit(nopython=True)(pyfunc)
    ret_c = cfunc(a_c)
    ret_f = cfunc(a_f)
    msg = 'No copy was performed, so the resulting array must not be writeable'
    self.assertTrue(not ret_c.flags.writeable, msg)
    msg = 'A copy was performed, yet the resulting array is not modifiable'
    self.assertTrue(ret_f.flags.writeable, msg)