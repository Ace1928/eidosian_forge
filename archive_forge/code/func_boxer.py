from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
@njit
def boxer():
    l = listobject.new_list(int32)
    for i in range(10, 20):
        l.append(i)
    return listobject._as_meminfo(l)