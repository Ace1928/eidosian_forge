import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def run_propagate_func(self, func, args):
    self.assertPreciseEqual(func(*args), func.py_func(*args))