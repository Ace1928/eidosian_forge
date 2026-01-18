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
def test_argsort_int(self):

    def check(pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        for orig in self.int_arrays():
            self.check_argsort(pyfunc, cfunc, orig)
    check(argsort_usecase)
    check(np_argsort_usecase)