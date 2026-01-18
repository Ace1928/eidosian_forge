import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer22():
    """Test to ensure that unsupported *args raises correctly"""

    def bar(a, b):
        pass
    x = (1, 2)
    bar(*x)