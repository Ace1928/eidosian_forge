from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
@overload(apply)
def ov_apply(array, func):
    return lambda array, func: func(array)