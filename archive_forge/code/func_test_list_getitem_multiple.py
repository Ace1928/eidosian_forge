from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_getitem_multiple(self):

    @njit
    def foo(i):
        l = listobject.new_list(int32)
        for j in range(10, 20):
            l.append(j)
        return l[i]
    for i, j in ((0, 10), (9, 19), (4, 14), (-5, 15), (-1, 19), (-10, 10)):
        self.assertEqual(foo(i), j)