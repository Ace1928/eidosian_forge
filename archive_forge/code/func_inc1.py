from numba import jit
import numpy as np
@jit
def inc1(a):
    a[0] += 1
    return a[0]