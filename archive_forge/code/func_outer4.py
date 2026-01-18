import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer4(x):
    """ Test recursive closure """
    y = x + 1

    def inner(x):
        if x + y < 2:
            return 10
        else:
            inner(x - 1)
    return inner(x)