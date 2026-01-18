import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def np_nditer2(a, b):
    res = []
    for u, v in np.nditer((a, b)):
        res.append((u.item(), v.item()))
    return res