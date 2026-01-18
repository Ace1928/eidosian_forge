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
def test_casefold(self):

    def pyfunc(x):
        return x.casefold()
    cfunc = njit(pyfunc)
    cpython = ['hello', 'hELlo', 'ß', 'ﬁ', 'Σ', 'AͅΣ', 'µ']
    cpython_extras = ['𐀀\U00100000']
    msg = 'Results of "{}".casefold() must be equal'
    for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
        self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))