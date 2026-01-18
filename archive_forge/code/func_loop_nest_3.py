import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def loop_nest_3(x, y):
    n = 0
    for i in range(x):
        for j in range(y):
            for k in range(x + y):
                n += i * j
    return n