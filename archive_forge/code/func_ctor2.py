import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
@njit
def ctor2():
    return dict(((1, 2), (3, 'a')))