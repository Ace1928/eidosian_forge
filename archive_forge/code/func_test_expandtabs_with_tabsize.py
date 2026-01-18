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
def test_expandtabs_with_tabsize(self):
    fns = [njit(expandtabs_with_tabsize_usecase), njit(expandtabs_with_tabsize_kwarg_usecase)]
    messages = ['Results of "{}".expandtabs({}) must be equal', 'Results of "{}".expandtabs(tabsize={}) must be equal']
    cases = ['', '\t', 't\tt\t', 'a\t', '\t⚡', 'a\tbc\nab\tc', '🐍\t⚡', '🐍⚡\n\t\t🐍\t', 'ab\rab\t\t\tab\r\n\ta']
    for s in cases:
        for tabsize in range(-1, 10):
            for fn, msg in zip(fns, messages):
                self.assertEqual(fn.py_func(s, tabsize), fn(s, tabsize), msg=msg.format(s, tabsize))