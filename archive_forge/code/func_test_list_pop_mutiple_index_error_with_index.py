from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_pop_mutiple_index_error_with_index(self):
    self.disable_leak_check()

    @njit
    def foo(i):
        l = listobject.new_list(int32)
        for j in (10, 11, 12):
            l.append(j)
        l.pop(i)
    with self.assertRaises(IndexError) as raises:
        foo(-4)
    self.assertIn('list index out of range', str(raises.exception))
    with self.assertRaises(IndexError) as raises:
        foo(3)
    self.assertIn('list index out of range', str(raises.exception))