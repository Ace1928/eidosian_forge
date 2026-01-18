from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_append_fails(self):
    self.disable_leak_check()

    @njit
    def foo():
        l = make_test_list()
        l._make_immutable()
        l.append(int32(1))
    with self.assertRaises(ValueError) as raises:
        foo()
    self.assertIn('list is immutable', str(raises.exception))