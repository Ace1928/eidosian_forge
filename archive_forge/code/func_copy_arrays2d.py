import math
import numpy as np
from numba import jit
def copy_arrays2d(a, b):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i, j] = a[i, j]