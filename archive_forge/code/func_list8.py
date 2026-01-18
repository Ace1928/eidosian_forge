import unittest
from numba.tests.support import TestCase
import sys
import operator
import numpy as np
import numpy
from numba import jit, njit, typed
from numba.core import types, utils
from numba.core.errors import TypingError, LoweringError
from numba.core.types.functions import _header_lead
from numba.np.numpy_support import numpy_version
from numba.tests.support import tag, _32bit, captured_stdout
def list8(x):
    """ Test use of list comprehension as arg to inner function """
    l = [z + 1 for z in x]

    def inner(x):
        return x[0] + 1
    q = inner(l)
    return q