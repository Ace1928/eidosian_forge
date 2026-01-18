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
def test_istitle(self):
    pyfunc = istitle_usecase
    cfunc = njit(pyfunc)
    error_msg = "'{0}'.py_istitle() = {1}\n'{0}'.c_istitle() = {2}"
    unicode_title = [x.title() for x in UNICODE_EXAMPLES]
    special = ['', '    ', '  AA  ', '  Ab  ', '1', 'A123', 'A12Bcd', '+abA', '12Abc', 'A12abc', '%^Abc 5 $% Def𐐁𐐩', '𐐧𐑎', '𐐩', '𐑎', '🐍 Is', '🐍 NOT', '👯Is', 'ῼ', 'Greek ῼitlecases ...']
    ISTITLE_EXAMPLES = UNICODE_EXAMPLES + unicode_title + special
    for s in ISTITLE_EXAMPLES:
        py_result = pyfunc(s)
        c_result = cfunc(s)
        self.assertEqual(py_result, c_result, error_msg.format(s, py_result, c_result))