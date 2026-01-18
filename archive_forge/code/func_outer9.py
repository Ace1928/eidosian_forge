import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer9(x):
    """ Test closure assignment"""
    z = x + 1

    def inner(x):
        return x + z
    f = inner
    return f(x)