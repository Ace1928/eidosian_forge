import platform
import numpy as np
from numba import types
import unittest
from numba import njit
from numba.core import config
from numba.tests.support import TestCase
def do_sum(x):
    acc = 0
    for v in np.nditer(x):
        acc += v.item()
    return acc