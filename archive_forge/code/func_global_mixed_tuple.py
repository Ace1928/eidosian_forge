import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def global_mixed_tuple():
    idx = tup_mixed[0]
    field = tup_mixed[1]
    return rec_X[idx][field]