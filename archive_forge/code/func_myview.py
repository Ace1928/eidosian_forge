import numpy as np
from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import TestCase
@njit
def myview():
    a = 1
    a.view(intty)