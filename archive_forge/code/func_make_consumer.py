import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def make_consumer(gen_func):

    def consumer(x):
        res = 0.0
        for y in gen_func(x):
            res += y
        return res
    return consumer