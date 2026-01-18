from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_singleton_delitem_index(self):

    @njit
    def foo():
        l = listobject.new_list(int32)
        l.append(0)
        del l[0]
        return len(l)
    self.assertEqual(foo(), 0)