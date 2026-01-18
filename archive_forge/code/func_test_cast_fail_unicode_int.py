from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_cast_fail_unicode_int(self):

    @njit
    def foo():
        l = listobject.new_list(int32)
        l.append('xyz')
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('cannot safely cast unicode_type to int32', str(raises.exception))