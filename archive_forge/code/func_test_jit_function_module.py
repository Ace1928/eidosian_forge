import unittest
from numba import jit
def test_jit_function_module(self):

    def add(x, y):
        return x + y
    c_add = jit(add)
    self.assertEqual(c_add.__module__, add.__module__)