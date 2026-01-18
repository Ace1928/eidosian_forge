import math
from numba import jit
from numba.core import types
from math import sqrt
import numpy as np
import numpy.random as nprand
@jit(forceobj=True)
def get_global_objmode(x):
    return K * x