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
def list16(x):
    """ Test type unification from np array ctors consuming list comprehension """
    l1 = [float(z) for z in x]
    l2 = [z for z in x]
    ze = np.array(l1)
    oe = np.array(l2)
    return ze + oe