from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_multiple_delitem(self):

    @njit
    def foo():
        l = listobject.new_list(int32)
        for j in (10, 11, 12):
            l.append(j)
        del l[0]
        return (len(l), l[0], l[1])
    self.assertEqual(foo(), (2, 11, 12))