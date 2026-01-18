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
def test_count_run(self):
    n = 16
    f = self.timsort.count_run

    def check(l, lo, hi):
        n, desc = f(self.array_factory(l), lo, hi)
        if desc:
            for k in range(lo, lo + n - 1):
                a, b = (l[k], l[k + 1])
                self.assertGreater(a, b)
            if lo + n < hi:
                self.assertLessEqual(l[lo + n - 1], l[lo + n])
        else:
            for k in range(lo, lo + n - 1):
                a, b = (l[k], l[k + 1])
                self.assertLessEqual(a, b)
            if lo + n < hi:
                self.assertGreater(l[lo + n - 1], l[lo + n], l)
    l = self.sorted_list(n, offset=100)
    check(l, 0, n)
    check(l, 1, n - 1)
    check(l, 1, 2)
    l = self.revsorted_list(n, offset=100)
    check(l, 0, n)
    check(l, 1, n - 1)
    check(l, 1, 2)
    l = self.random_list(n, offset=100)
    for i in range(len(l) - 1):
        check(l, i, n)
    l = self.duprandom_list(n, offset=100)
    for i in range(len(l) - 1):
        check(l, i, n)