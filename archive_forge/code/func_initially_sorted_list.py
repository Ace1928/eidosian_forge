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
def initially_sorted_list(self, n, m=None, offset=10):
    if m is None:
        m = n // 2
    l = self.sorted_list(m, offset)
    l += self.random_list(n - m, offset=l[-1] + offset)
    return l