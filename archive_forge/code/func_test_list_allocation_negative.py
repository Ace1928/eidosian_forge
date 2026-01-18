from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_allocation_negative(self):

    @njit
    def foo():
        l = listobject.new_list(int32, -1)
        return l._allocated()
    with self.assertRaises(RuntimeError) as raises:
        self.assertEqual(foo(), -1)
    self.assertIn('expecting *allocated* to be >= 0', str(raises.exception))