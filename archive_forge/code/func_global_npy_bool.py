import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def global_npy_bool():
    _sink(_glbl_np_bool_T, _glbl_np_bool_F)
    return (_glbl_np_bool_T, _glbl_np_bool_F)