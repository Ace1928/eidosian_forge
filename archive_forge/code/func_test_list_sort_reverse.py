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
def test_list_sort_reverse(self):
    pyfunc = list_sort_reverse_usecase
    cfunc = jit(nopython=True)(pyfunc)
    for size in (20, 50, 500):
        for b in (False, True):
            orig, ret = cfunc(size, b)
            self.assertEqual(sorted(orig, reverse=b), ret)
            self.assertNotEqual(orig, ret)