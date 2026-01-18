import re
from numba import njit
from numba.core.extending import overload
from numba.core.targetconfig import ConfigStack
from numba.core.compiler import Flags, DEFAULT_FLAGS
from numba.core import types
from numba.core.funcdesc import default_mangler
from numba.tests.support import TestCase, unittest
def test_fastmath_in_overload(self):

    def fastmath_status():
        pass

    @overload(fastmath_status)
    def ov_fastmath_status():
        flags = ConfigStack().top()
        val = 'Has fastmath' if flags.fastmath else 'No fastmath'

        def codegen():
            return val
        return codegen

    @njit(fastmath=True)
    def set_fastmath():
        return fastmath_status()

    @njit()
    def foo():
        a = fastmath_status()
        b = set_fastmath()
        return (a, b)
    a, b = foo()
    self.assertEqual(a, 'No fastmath')
    self.assertEqual(b, 'Has fastmath')