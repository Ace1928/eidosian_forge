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
def test_eq_optional(self):

    @njit
    def foo(pred1, pred2):
        if pred1 > 0:
            resolved1 = 'concrete'
        else:
            resolved1 = None
        if pred2 < 0:
            resolved2 = 'concrete'
        else:
            resolved2 = None
        if resolved1 == resolved2:
            return 10
        else:
            return 20
    for p1, p2 in product(*((-1, 1),) * 2):
        self.assertEqual(foo(p1, p2), foo.py_func(p1, p2))