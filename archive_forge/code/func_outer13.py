import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer13(x, kw=7):
    """ outer with kwarg no closure"""
    z = x + 1 + kw
    return z