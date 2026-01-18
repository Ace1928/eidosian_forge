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
def test_expandtabs(self):
    pyfunc = expandtabs_usecase
    cfunc = njit(pyfunc)
    cases = ['', '\t', 't\tt\t', 'a\t', '\t⚡', 'a\tbc\nab\tc', '🐍\t⚡', '🐍⚡\n\t\t🐍\t', 'ab\rab\t\t\tab\r\n\ta']
    msg = 'Results of "{}".expandtabs() must be equal'
    for s in cases:
        self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))