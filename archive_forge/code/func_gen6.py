import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def gen6(a, b):
    x = a + 1
    while True:
        y = b + 2
        yield (x + y)