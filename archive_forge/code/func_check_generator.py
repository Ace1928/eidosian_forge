import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def check_generator(self, pygen, cgen):
    self.assertEqual(next(cgen), next(pygen))
    expected = [x for x in pygen]
    got = [x for x in cgen]
    self.assertEqual(expected, got)
    with self.assertRaises(StopIteration):
        next(cgen)