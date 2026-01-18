import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def np_nditer1(a):
    res = []
    for u in np.nditer(a):
        res.append(u.item())
    return res