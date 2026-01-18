import math
import numpy as np
from numba import jit
def redefine1():
    x = 0
    for i in range(5):
        x += 1
    x = 0.0 + x
    for i in range(5):
        x += 1
    return x