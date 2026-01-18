from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_copy_multiple(self):

    @njit
    def foo():
        l = listobject.new_list(int32)
        for j in range(10, 13):
            l.append(j)
        n = l.copy()
        return (len(l), len(n), l[0], l[1], l[2], l[0], l[1], l[2])
    self.assertEqual(foo(), (3, 3, 10, 11, 12, 10, 11, 12))