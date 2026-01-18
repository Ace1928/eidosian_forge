from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_multiple_delitem_off_by_one(self):

    @njit
    def foo():
        l = listobject.new_list(int32)
        for j in range(10, 20):
            l.append(j)
        k = listobject.new_list(int32)
        for j in range(10, 20):
            k.append(j)
        del l[-9:-20]
        return k == l
    self.assertTrue(foo())