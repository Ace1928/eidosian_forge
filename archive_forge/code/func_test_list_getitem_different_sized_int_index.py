from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_getitem_different_sized_int_index(self):
    for ty in types.signed_domain:

        @njit
        def foo():
            l = listobject.new_list(int32)
            l.append(7)
            return (l[ty(0)], l[ty(-1)])
        self.assertEqual(foo(), (7, 7))