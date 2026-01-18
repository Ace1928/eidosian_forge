import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer10(x):
    """ Test two inner, one calls other """
    z = x + 1

    def inner(x):
        return x + z

    def inner2(x):
        return inner(x)
    return inner2(x)