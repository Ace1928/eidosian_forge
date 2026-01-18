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
def test_find(self, flags=no_pyobj_flags):
    pyfunc = find_usecase
    cfunc = njit(pyfunc)
    default_subs = [(s, ['', 'xx', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES]
    cpython_subs = [('a' * 100 + 'Ă', ['Ă', 'ȁ', 'Ġ', 'Ƞ']), ('a' * 100 + '\U00100304', ['\U00100304', '\U00100204', '\U00102004']), ('Ă' * 100 + 'a', ['a']), ('\U00100304' * 100 + 'a', ['a']), ('\U00100304' * 100 + 'Ă', ['Ă']), ('a' * 100, ['Ă', '\U00100304', 'aĂ', 'a\U00100304']), ('Ă' * 100, ['\U00100304', 'Ă\U00100304']), ('Ă' * 100 + 'a_', ['a_']), ('\U00100304' * 100 + 'a_', ['a_']), ('\U00100304' * 100 + 'Ă_', ['Ă_'])]
    for s, subs in default_subs + cpython_subs:
        for sub_str in subs:
            msg = 'Results "{}".find("{}") must be equal'
            self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str), msg=msg.format(s, sub_str))