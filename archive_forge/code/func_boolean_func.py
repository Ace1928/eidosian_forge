import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba import cuda
def boolean_func(A, vertial):
    if vertial:
        A[0] = 123
    else:
        A[0] = 321