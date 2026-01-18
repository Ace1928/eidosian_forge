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
def list17(x):
    """ Test complex list comprehension including math calls """
    return [(a, b, c) for a in x for b in x for c in x if np.sqrt(a ** 2 + b ** 2) == c]