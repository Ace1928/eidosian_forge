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
def list22(x):
    """ Test create two lists comprehensions and a third walking the first two """
    a = [y - 1 for y in x]
    b = [y + 1 for y in x]
    return [x for x in a for y in b if x == y]