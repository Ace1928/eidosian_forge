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
def test_f_strings(self):
    """test f-string support, which requires bytecode handling
        """

    def impl1(a):
        return f'AA_{a + 3}_B'

    def impl2(a):
        return f'{a + 2}'

    def impl3(a):
        return f'ABC_{a}'

    def impl4(a):
        return f'ABC_{a:0}'

    def impl5():
        return f''
    self.assertEqual(impl1(3), njit(impl1)(3))
    self.assertEqual(impl2(2), njit(impl2)(2))
    self.assertEqual(impl3('DE'), njit(impl3)('DE'))
    list_arg = ['A', 'B']
    got = njit(impl3)(list_arg)
    expected = f'ABC_<object type:{typeof(list_arg)}>'
    self.assertEqual(got, expected)
    with self.assertRaises(UnsupportedError) as raises:
        njit(impl4)(['A', 'B'])
    msg = 'format spec in f-strings not supported yet'
    self.assertIn(msg, str(raises.exception))
    self.assertEqual(impl5(), njit(impl5)())