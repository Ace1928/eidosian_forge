from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
import numpy as np
@cuda.jit
def enumerator(x, error):
    count = 0
    for i, v in enumerate(x):
        if count != i:
            error[0] = 1
        if v != x[i]:
            error[0] = 2
        count += 1
    if count != len(x):
        error[0] = 3