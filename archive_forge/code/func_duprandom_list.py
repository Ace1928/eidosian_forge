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
def duprandom_list(self, n, factor=None, offset=10):
    random.seed(42)
    if factor is None:
        factor = int(math.sqrt(n))
    l = (list(range(offset, offset + n // factor + 1)) * (factor + 1))[:n]
    assert len(l) == n
    random.shuffle(l)
    return l