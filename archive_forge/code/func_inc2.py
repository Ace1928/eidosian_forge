from numba import jit
import numpy as np
@jit
def inc2(a):
    a[0] += 1
    return (a[0], a[0] + 1)