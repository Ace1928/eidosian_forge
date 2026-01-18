import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
@njit
def ctor3():
    return dict((('a', 'b', 'c'), ('d', 'e', 'f')))