from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_pop_multiple_index(self):

    @njit
    def foo(i):
        l = listobject.new_list(int32)
        for j in (10, 11, 12):
            l.append(j)
        return (l.pop(i), len(l))
    for i, n in ((0, 10), (1, 11), (2, 12)):
        self.assertEqual(foo(i), (n, 2))
    for i, n in ((-3, 10), (-2, 11), (-1, 12)):
        self.assertEqual(foo(i), (n, 2))