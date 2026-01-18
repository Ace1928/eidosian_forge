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
def test_find_with_start_only(self):
    pyfunc = find_with_start_only_usecase
    cfunc = njit(pyfunc)
    for s in UNICODE_EXAMPLES:
        for sub_str in ['', 'xx', s[:-2], s[3:], s]:
            for start in list(range(-20, 20)) + [None]:
                msg = 'Results "{}".find("{}", {}) must be equal'
                self.assertEqual(pyfunc(s, sub_str, start), cfunc(s, sub_str, start), msg=msg.format(s, sub_str, start))