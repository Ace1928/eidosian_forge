import numpy as np
from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import TestCase
def gen_view(a, b):

    def impl(x):
        return a(x).view(b)
    return impl