import numpy as np
import unittest
from numba import jit, njit
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.tests.support import no_pyobj_flags as nullary_no_pyobj_flags
from numba.tests.support import force_pyobj_flags as nullary_force_pyobj_flags
def check_unpack_error(self, pyfunc, flags=force_pyobj_flags, exc=ValueError):
    with self.assertRaises(exc):
        cfunc = jit((), **flags)(pyfunc)
        cfunc()