import unittest
from numba import jit
def test_jit_function_code_object(self):

    def add(x, y):
        return x + y
    c_add = jit(add)
    self.assertEqual(c_add.__code__, add.__code__)
    self.assertEqual(c_add.func_code, add.__code__)