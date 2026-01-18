import math
import numpy as np
from numba import jit
def string_comparison(s1, s2, op):
    return op(s1, s2)