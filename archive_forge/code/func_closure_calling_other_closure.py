import math
from numba import jit
from numba.core import types
from math import sqrt
import numpy as np
import numpy.random as nprand
def closure_calling_other_closure(x):

    @jit(nopython=True)
    def other_inner(y):
        return math.hypot(x, y)

    @jit(nopython=True)
    def inner(y):
        return other_inner(y) + x
    return inner