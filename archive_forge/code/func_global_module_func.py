import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
@jit(nopython=True)
def global_module_func(x, y):
    return usecases.andornopython(x, y)