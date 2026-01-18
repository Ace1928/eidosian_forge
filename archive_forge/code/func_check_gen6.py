import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def check_gen6(self, **kwargs):
    cr = jit((types.int32,) * 2, **kwargs)(gen6)
    cgen = cr(5, 6)
    l = []
    for i in range(3):
        l.append(next(cgen))
    self.assertEqual(l, [14] * 3)