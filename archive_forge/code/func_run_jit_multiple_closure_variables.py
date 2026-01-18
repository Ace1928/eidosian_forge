import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def run_jit_multiple_closure_variables(self, **jitargs):
    Y = 10
    Z = 2

    def add_Y_mult_Z(x):
        return (x + Y) * Z
    c_add_Y_mult_Z = jit('i4(i4)', **jitargs)(add_Y_mult_Z)
    self.assertEqual(c_add_Y_mult_Z(1), 22)