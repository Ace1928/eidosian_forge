import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def check_np_flat(self, pyfunc, **kwargs):
    cr = jit((types.Array(types.int32, 2, 'C'),), **kwargs)(pyfunc)
    arr = np.arange(6, dtype=np.int32).reshape((2, 3))
    self.check_generator(pyfunc(arr), cr(arr))
    crA = jit((types.Array(types.int32, 2, 'A'),), **kwargs)(pyfunc)
    arr = arr.T
    self.check_generator(pyfunc(arr), crA(arr))