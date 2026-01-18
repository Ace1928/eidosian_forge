import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def array_flat_sum(arr):
    s = 0
    for i, v in enumerate(arr.flat):
        s = s + (i + 1) * v
    return s