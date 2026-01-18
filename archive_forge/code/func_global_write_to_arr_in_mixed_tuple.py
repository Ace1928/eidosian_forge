import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def global_write_to_arr_in_mixed_tuple():
    mixed_tup_tup_array[0][1][0] = 10.0