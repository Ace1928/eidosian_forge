import math
from numba import jit
from numba.core import types
from math import sqrt
import numpy as np
import numpy.random as nprand
@jit
def add_without_sig(a, b):
    return a + b