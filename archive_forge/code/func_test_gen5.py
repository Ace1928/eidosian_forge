import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_gen5(self):
    with self.assertTypingError() as raises:
        jit((), **nopython_flags)(gen5)
    self.assertIn('Cannot type generator: it does not yield any value', str(raises.exception))