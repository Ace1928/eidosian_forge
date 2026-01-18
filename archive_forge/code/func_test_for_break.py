import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_for_break(self, flags=force_pyobj_jit_opt):
    pyfunc = for_break
    cfunc = jit((types.intp, types.intp), **flags)(pyfunc)
    for n, x in [(4, 2), (4, 6)]:
        self.assertPreciseEqual(pyfunc(n, x), cfunc(n, x))