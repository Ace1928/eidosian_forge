import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def check_gen7(self, **kwargs):
    pyfunc = gen7
    cr = jit((types.Array(types.float64, 1, 'C'),), **kwargs)(pyfunc)
    arr = np.linspace(1, 10, 7)
    pygen = pyfunc(arr.copy())
    cgen = cr(arr)
    self.check_generator(pygen, cgen)