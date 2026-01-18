import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
@njit
def make_optional(val, get_none):
    if get_none:
        ret = None
    else:
        ret = val
    return ret