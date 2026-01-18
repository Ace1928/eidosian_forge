from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_atleast_2d(self):
    pyfunc = atleast_2d
    cfunc = jit(nopython=True)(pyfunc)
    self.check_atleast_nd(pyfunc, cfunc)