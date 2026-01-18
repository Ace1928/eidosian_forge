import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def gen8(x=1, y=2, b=False):
    bb = not b
    yield x
    if bb:
        yield y
    if b:
        yield (x + y)