import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer20(x):
    """ Test calling numpy in closure """
    z = x + 1

    def inner(x):
        return x + numpy.cos(z)
    return inner(x)