from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_setitem_integer_types_as_index(self):

    @njit
    def foo(i):
        l = listobject.new_list(int32)
        l.append(0)
        l[i] = 1
        return l[i]
    for t in types.signed_domain:
        self.assertEqual(foo(t(0)), 1)