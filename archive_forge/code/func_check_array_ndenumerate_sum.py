import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def check_array_ndenumerate_sum(self, arr, arrty):
    self.check_array_unary(arr, arrty, array_ndenumerate_sum)