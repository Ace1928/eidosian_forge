import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
def fancy_array_modify(x):
    a = np.array([1, 2, 3])
    x[a] = 0
    return x