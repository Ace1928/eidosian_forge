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
def test_islower(self):
    pyfunc = islower_usecase
    cfunc = njit(pyfunc)
    lowers = [x.lower() for x in UNICODE_EXAMPLES]
    extras = ['AA12A', 'aa12a', '大AA12A', '大aa12a', 'AAAǄA', 'A 1 1 大']
    cpython = ['Ⅷ', 'ⅷ', '𐐁', '𐐧', '𐐩', '𐑎', '🐍', '👯']
    cpython += [x * 4 for x in cpython]
    msg = 'Results of "{}".islower() must be equal'
    for s in UNICODE_EXAMPLES + lowers + [''] + extras + cpython:
        self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))