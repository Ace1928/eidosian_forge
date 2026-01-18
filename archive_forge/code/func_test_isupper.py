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
def test_isupper(self):

    def pyfunc(x):
        return x.isupper()
    cfunc = njit(pyfunc)
    uppers = [x.upper() for x in UNICODE_EXAMPLES]
    extras = ['AA12A', 'aa12a', '大AA12A', '大aa12a', 'AAAǄA', 'A 1 1 大']
    cpython = ['Ⅷ', 'ⅷ', '𐐁', '𐐧', '𐐩', '𐑎', '🐍', '👯']
    fourxcpy = [x * 4 for x in cpython]
    for a in UNICODE_EXAMPLES + uppers + extras + cpython + fourxcpy:
        args = [a]
        self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))