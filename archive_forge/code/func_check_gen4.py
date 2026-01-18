import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def check_gen4(self, **kwargs):
    pyfunc = gen4
    cr = jit((types.int32,) * 3, **kwargs)(pyfunc)
    pygen = pyfunc(5, 6, 7)
    cgen = cr(5, 6, 7)
    self.check_generator(pygen, cgen)