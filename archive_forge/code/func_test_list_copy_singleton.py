from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_copy_singleton(self):

    @njit
    def foo():
        l = listobject.new_list(int32)
        l.append(0)
        n = l.copy()
        return (len(l), len(n), l[0], n[0])
    self.assertEqual(foo(), (1, 1, 0, 0))