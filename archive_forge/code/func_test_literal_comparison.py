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
def test_literal_comparison(self):

    def pyfunc(option):
        x = 'a123'
        y = 'aa12'
        if option == '==':
            return x == y
        elif option == '!=':
            return x != y
        elif option == '<':
            return x < y
        elif option == '>':
            return x > y
        elif option == '<=':
            return x <= y
        elif option == '>=':
            return x >= y
        else:
            return None
    cfunc = njit(pyfunc)
    for cmpop in ['==', '!=', '<', '>', '<=', '>=', '']:
        args = [cmpop]
        self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))