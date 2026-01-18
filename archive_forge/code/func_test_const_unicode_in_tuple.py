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
def test_const_unicode_in_tuple(self):

    @njit
    def f():
        return ('aa',) < ('bb',)
    self.assertEqual(f.py_func(), f())

    @njit
    def f():
        return ('cc',) < ('bb',)
    self.assertEqual(f.py_func(), f())