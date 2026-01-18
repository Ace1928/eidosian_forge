import math
from numba import jit
from numba.core import types
from math import sqrt
import numpy as np
import numpy.random as nprand
def closure_calling_other_function(x):

    @jit(nopython=True)
    def inner(y, z):
        return other_function(x, y) + z
    return inner