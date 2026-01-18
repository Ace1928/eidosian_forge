import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def global_str_tuple():
    return tup_str[0] + tup_str[1]