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
def test_ascii_flag_join(self):

    @njit
    def f():
        s1 = 'abc'
        s2 = '123'
        s3 = '🐍⚡'
        s4 = '大处着眼，小处着手。'
        return (','.join([s1, s2])._is_ascii, '🐍⚡'.join([s1, s2])._is_ascii, ','.join([s1, s3])._is_ascii, ','.join([s3, s4])._is_ascii)
    self.assertEqual(f(), (1, 0, 0, 0))