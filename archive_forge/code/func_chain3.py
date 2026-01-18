from numba import jit
import numpy as np
def chain3(x, y):
    a = b, c = (inc1(x), inc1(y))
    d, e = f = (inc1(x), inc1(y))
    return (a[0] + b / 2 + d + f[0], a[1] + c + e / 2 + f[1])