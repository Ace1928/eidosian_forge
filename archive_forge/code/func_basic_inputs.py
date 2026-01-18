import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def basic_inputs(self):
    yield np.arange(4).astype(np.complex64)
    yield np.arange(8)[::2]
    a = np.arange(12).reshape((3, 4))
    yield a
    yield a.copy(order='F')