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
def test_ascii_flag_getitem(self):

    @njit
    def f():
        s1 = 'abc123'
        s2 = '🐍⚡🐍⚡🐍⚡'
        return (s1[0]._is_ascii, s1[2:]._is_ascii, s2[0]._is_ascii, s2[2:]._is_ascii)
    self.assertEqual(f(), (1, 1, 0, 0))