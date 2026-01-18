import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def nested_enumerate_usecase():
    res = 0.0
    for i, (j, k) in enumerate(enumerate(((1.5, 2.0), (99.3, 3.4), (1.8, 2.5)))):
        for l in k:
            res += i * j * l
        res = res * 2
    return res