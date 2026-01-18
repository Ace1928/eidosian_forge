import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_assignments2(self, flags=force_pyobj_jit_opt):
    pyfunc = assignments2
    cfunc = jit((types.int32,), **flags)(pyfunc)
    for x in [-1, 0, 1]:
        self.assertPreciseEqual(pyfunc(x), cfunc(x))
    if flags is force_pyobj_jit_opt:
        cfunc('a')