import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def return_double_or_none(x):
    if x:
        ret = None
    else:
        ret = 1.2
    return ret