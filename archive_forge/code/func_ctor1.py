import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
@njit
def ctor1():
    d = dict()
    d[1] = 2
    return dict(d)