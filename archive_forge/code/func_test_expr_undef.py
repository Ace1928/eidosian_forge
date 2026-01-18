import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def test_expr_undef(self):

    @jit(forceobj=True)
    def foo():
        return [x for x in (1, 2)]
    self.assertEqual(foo(), foo.py_func())