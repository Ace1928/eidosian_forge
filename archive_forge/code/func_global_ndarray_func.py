import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def global_ndarray_func(x):
    y = x + X.shape[0]
    return y