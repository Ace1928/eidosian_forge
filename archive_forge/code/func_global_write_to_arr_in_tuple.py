import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def global_write_to_arr_in_tuple():
    tup_tup_array[0][0][0] = 10.0