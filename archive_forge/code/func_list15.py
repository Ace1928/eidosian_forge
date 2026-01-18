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
def list15(x):
    """ Test use of python scalar ctor reference in list comprehension followed by np array construction from the list"""
    l = [float(z) for z in x]
    return np.array(l)