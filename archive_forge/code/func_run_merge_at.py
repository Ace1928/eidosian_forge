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
def run_merge_at(ms, keys, i):
    new_ms = f(ms, keys, keys, i)
    self.assertEqual(keys[0], orig_keys[0])
    self.assertEqual(keys[-1], orig_keys[-1])
    self.assertSorted(orig_keys[1:-1], keys[1:-1])
    self.assertIs(new_ms.pending, ms.pending)
    self.assertEqual(ms.pending[i], (ssa, na + nb))
    self.assertEqual(ms.pending[0], stack_sentinel)
    return new_ms