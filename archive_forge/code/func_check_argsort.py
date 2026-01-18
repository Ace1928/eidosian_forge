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
def check_argsort(self, pyfunc, cfunc, val, kwargs={}):
    orig = copy.copy(val)
    expected = pyfunc(val, **kwargs)
    got = cfunc(val, **kwargs)
    self.assertPreciseEqual(orig[got], np.sort(orig), msg="the array wasn't argsorted")
    if not self.has_duplicates(orig):
        self.assertPreciseEqual(got, expected)
    self.assertPreciseEqual(val, orig)