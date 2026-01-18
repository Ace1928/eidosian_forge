from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_iter_self_mutation(self):
    self.disable_leak_check()

    @njit
    def foo():
        l = listobject.new_list(int32)
        l.extend((1, 2, 3, 4))
        for i in l:
            l.append(i)
    with self.assertRaises(RuntimeError) as raises:
        foo()
    self.assertIn('list was mutated during iteration'.format(**locals()), str(raises.exception))