import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def gen2(x):
    for i in range(x):
        yield i
        for j in range(1, 3):
            yield (i + j)