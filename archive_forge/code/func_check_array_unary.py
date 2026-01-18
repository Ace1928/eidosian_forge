import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def check_array_unary(self, arr, arrty, func):
    cfunc = njit((arrty,))(func)
    self.assertPreciseEqual(cfunc(arr), func(arr))