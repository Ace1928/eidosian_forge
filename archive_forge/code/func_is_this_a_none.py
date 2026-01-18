import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def is_this_a_none(x):
    if x:
        val_or_none = None
    else:
        val_or_none = x
    if val_or_none is None:
        return x - 1
    if val_or_none is not None:
        return x + 1