import unittest
from numba import jit
def test_jit_function_docstring(self):

    def add(x, y):
        """Return sum of two numbers"""
        return x + y
    c_add = jit(add)
    self.assertEqual(c_add.__doc__, 'Return sum of two numbers')