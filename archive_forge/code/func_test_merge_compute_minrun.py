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
def test_merge_compute_minrun(self):
    f = self.timsort.merge_compute_minrun
    for i in range(0, 64):
        self.assertEqual(f(i), i)
    for i in range(6, 63):
        if 2 ** i > sys.maxsize:
            break
        self.assertEqual(f(2 ** i), 32)
    for i in self.fibo():
        if i < 64:
            continue
        if i >= sys.maxsize:
            break
        k = f(i)
        self.assertGreaterEqual(k, 32)
        self.assertLessEqual(k, 64)
        if i > 500:
            quot = i // k
            p = 2 ** utils.bit_length(quot)
            self.assertLess(quot, p)
            self.assertGreaterEqual(quot, 0.9 * p)