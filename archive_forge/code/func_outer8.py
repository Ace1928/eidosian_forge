import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer8(x):
    """ Test use of outer scope var, with closure """
    z = x + 1

    def inner(x):
        return x + z + _OUTER_SCOPE_VAR
    return inner(x)