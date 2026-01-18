from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_multiple_not_equal(self):

    @njit
    def foo():
        t = listobject.new_list(int32)
        o = listobject.new_list(int32)
        for i in range(10):
            t.append(i)
            o.append(i)
        o[-1] = 42
        return (t == o, t != o)
    self.assertEqual(foo(), (False, True))