import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_build_unpack_fail_on_list_assign_like(self):

    def check(p):
        pyfunc = lambda a: (*a,)
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(p), pyfunc(p))
    with self.assertRaises(errors.TypingError) as raises:
        check([4, 5])
    msg1 = 'No implementation of function'
    self.assertIn(msg1, str(raises.exception))
    msg2 = 'tuple(reflected list('
    self.assertIn(msg2, str(raises.exception))