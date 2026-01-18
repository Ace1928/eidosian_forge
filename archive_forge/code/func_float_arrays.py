import copy
import itertools
import math
import random
import sys
import unittest
import numpy as np
from numba import jit, njit
from numba.core import utils, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.misc.quicksort import make_py_quicksort, make_jit_quicksort
from numba.misc.mergesort import make_jit_mergesort
from numba.misc.timsort import make_py_timsort, make_jit_timsort, MergeRun
def float_arrays(self):
    for size in (5, 20, 50, 500):
        yield (np.random.random(size=size) * 100)
    for size in (5, 20, 50, 500):
        orig = np.random.random(size=size) * 100
        orig[np.random.random(size=size) < 0.1] = float('nan')
        yield orig
    for size in (50, 500):
        orig = np.random.random(size=size) * 100
        orig[np.random.random(size=size) < 0.9] = float('nan')
        yield orig