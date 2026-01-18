from itertools import product
from itertools import permutations
from numba import njit, typeof
from numba.core import types
import unittest
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.core.errors import TypingError, UnsupportedError
from numba.cpython.unicode import _MAX_UNICODE
from numba.core.types.functions import _header_lead
from numba.extending import overload
def test_isascii(self):

    def pyfunc(x):
        return x.isascii()
    cfunc = njit(pyfunc)
    cpython = ['', '\x00', '\x7f', '\x00\x7f', '\x80', 'é', ' ']
    msg = 'Results of "{}".isascii() must be equal'
    for s in UNICODE_EXAMPLES + cpython:
        self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))