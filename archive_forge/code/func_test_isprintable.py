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
def test_isprintable(self):

    def pyfunc(s):
        return s.isprintable()
    cfunc = njit(pyfunc)
    cpython = ['', ' ', 'abcdefg', 'abcdefg\n', 'ʹ', '\u0378', '\ud800', '👯', '\U000e0020']
    msg = 'Results of "{}".isprintable() must be equal'
    for s in UNICODE_EXAMPLES + cpython:
        self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))