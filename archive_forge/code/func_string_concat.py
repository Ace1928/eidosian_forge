import math
import numpy as np
from numba import jit
def string_concat(x, y):
    a = 'whatzup'
    return a + str(x + y)