import platform
import numpy as np
from numba import types
import unittest
from numba import njit
from numba.core import config
from numba.tests.support import TestCase
def sum_sqrt_list(lst):
    acc = 0.0
    for item in lst:
        acc += np.sqrt(item)
    return acc