import re
from numba import njit
from numba.core.extending import overload
from numba.core.targetconfig import ConfigStack
from numba.core.compiler import Flags, DEFAULT_FLAGS
from numba.core import types
from numba.core.funcdesc import default_mangler
from numba.tests.support import TestCase, unittest
@overload(fastmath_status)
def ov_fastmath_status():
    flags = ConfigStack().top()
    val = 'Has fastmath' if flags.fastmath else 'No fastmath'

    def codegen():
        return val
    return codegen