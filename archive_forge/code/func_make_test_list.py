from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
@register_jitable
def make_test_list():
    l = listobject.new_list(int32)
    l.append(int32(1))
    return l