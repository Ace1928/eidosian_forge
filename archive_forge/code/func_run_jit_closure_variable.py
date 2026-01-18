import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def run_jit_closure_variable(self, **jitargs):
    Y = 10

    def add_Y(x):
        return x + Y
    c_add_Y = jit('i4(i4)', **jitargs)(add_Y)
    self.assertEqual(c_add_Y(1), 11)
    Y = 12
    self.assertEqual(c_add_Y(1), 11)