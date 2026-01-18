from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_setitem_multiple_index_error(self):
    self.disable_leak_check()

    @njit
    def foo(i):
        l = listobject.new_list(int32)
        for j in range(10, 20):
            l.append(j)
        l[i] = 0
    with self.assertRaises(IndexError):
        foo(10)
    with self.assertRaises(IndexError):
        foo(-11)