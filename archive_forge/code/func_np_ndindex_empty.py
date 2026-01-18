import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def np_ndindex_empty():
    s = 0
    for ind in np.ndindex(()):
        s += s + len(ind) + 1
    return s