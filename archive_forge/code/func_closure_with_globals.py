import math
from numba import jit
from numba.core import types
from math import sqrt
import numpy as np
import numpy.random as nprand
def closure_with_globals(x, **jit_args):

    @jit(**jit_args)
    def inner(y):
        k = max(K, K + 1)
        return math.hypot(x, y) + sqrt(k)
    return inner