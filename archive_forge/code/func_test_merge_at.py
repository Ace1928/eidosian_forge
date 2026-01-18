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
def test_merge_at(self):
    for na, nb in [(12, 16), (40, 40), (100, 110), (500, 510)]:
        for a, b in itertools.product(self.make_sample_sorted_lists(na), self.make_sample_sorted_lists(nb)):
            self.check_merge_at(a, b)
            self.check_merge_at(b, a)