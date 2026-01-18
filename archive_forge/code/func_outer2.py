import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer2(x):
    """ Test calling recursive function from closure """
    z = x + 1

    def inner(x):
        return x + fib3(z)
    return inner(x)