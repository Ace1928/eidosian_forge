import numpy as np
import unittest
from numba import jit, njit
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.tests.support import no_pyobj_flags as nullary_no_pyobj_flags
from numba.tests.support import force_pyobj_flags as nullary_force_pyobj_flags
def test_chained_unpack_assign(self, flags=force_pyobj_flags):
    pyfunc = chained_unpack_assign1
    cfunc = jit((types.int32, types.int32), **flags)(pyfunc)
    args = (4, 5)
    self.assertPreciseEqual(cfunc(*args), pyfunc(*args))