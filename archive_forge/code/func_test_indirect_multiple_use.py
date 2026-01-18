import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def test_indirect_multiple_use(self):
    """
        Issue #2263

        Linkage error due to multiple definition of global tracking symbol.
        """
    my_sin = mod.cffi_sin

    @jit(nopython=True)
    def inner(x):
        return my_sin(x)

    @jit(nopython=True)
    def foo(x):
        return inner(x) + my_sin(x + 1)
    x = 1.123
    self.assertEqual(foo(x), my_sin(x) + my_sin(x + 1))