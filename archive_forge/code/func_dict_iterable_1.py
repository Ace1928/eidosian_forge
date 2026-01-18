import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
@njit
def dict_iterable_1(a, b):
    d = dict(zip(a, b))
    return d