import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_gen5_objmode(self):
    cgen = jit((), **forceobj_flags)(gen5)()
    self.assertEqual(list(cgen), [])
    with self.assertRaises(StopIteration):
        next(cgen)