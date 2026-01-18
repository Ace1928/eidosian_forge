from numba import jit
import numpy as np
def chain1(a):
    x = y = z = inc(a)
    return x + y + z