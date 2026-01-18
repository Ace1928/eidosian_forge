from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_pop_singleton_typing_error_on_index(self):
    self.disable_leak_check()

    @njit
    def foo(i):
        l = listobject.new_list(int32)
        l.append(0)
        return l.pop(i)
    for i in ('xyz', 1.0, 1j):
        with self.assertRaises(TypingError) as raises:
            foo(i)
        self.assertIn('argument for pop must be an integer', str(raises.exception))