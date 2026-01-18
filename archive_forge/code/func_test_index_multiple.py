from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_index_multiple(self):

    @njit
    def foo(i):
        l = listobject.new_list(int32)
        for j in range(10, 20):
            l.append(j)
        return l.index(i)
    for i, v in zip(range(10), range(10, 20)):
        self.assertEqual(foo(v), i)