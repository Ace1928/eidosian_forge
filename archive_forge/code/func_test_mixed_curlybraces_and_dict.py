import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_mixed_curlybraces_and_dict(self):

    @njit
    def foo():
        k = dict()
        k[1] = {1: 3}
        k[2] = {4: 2}
        return k
    self.assertEqual(foo(), foo.py_func())