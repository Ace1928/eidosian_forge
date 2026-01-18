import sys
import numpy as np
from numba import njit
from numba.tests.support import TestCase
@njit(parallel=True, cache=True)
def arrayexprs_case(arr):
    return arr / arr.sum()