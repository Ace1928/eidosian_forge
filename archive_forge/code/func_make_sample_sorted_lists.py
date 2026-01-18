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
def make_sample_sorted_lists(self, n):
    lists = []
    for offset in (20, 120):
        lists.append(self.sorted_list(n, offset))
        lists.append(self.dupsorted_list(n, offset))
    return lists