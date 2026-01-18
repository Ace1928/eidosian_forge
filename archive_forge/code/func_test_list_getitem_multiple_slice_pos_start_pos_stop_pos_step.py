from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_getitem_multiple_slice_pos_start_pos_stop_pos_step(self):

    @njit
    def foo():
        l = listobject.new_list(int32)
        for j in range(10, 20):
            l.append(j)
        n = l[1:9:2]
        return (len(n), (n[0], n[1], n[2], n[3]))
    length, items = foo()
    self.assertEqual(length, 4)
    self.assertEqual(items, (11, 13, 15, 17))