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
def test_merge_force_collapse(self):
    f = self.timsort.merge_force_collapse
    sizes_list = [(8, 10, 15, 20)]
    sizes_list.append(sizes_list[0][::-1])
    for sizes in sizes_list:
        for chunks in itertools.product(*(self.make_sample_sorted_lists(n) for n in sizes)):
            orig_keys = sum(chunks, [])
            keys = self.array_factory(orig_keys)
            ms = self.merge_init(keys)
            pos = 0
            for c in chunks:
                ms = self.timsort.merge_append(ms, MergeRun(pos, len(c)))
                pos += len(c)
            self.assertEqual(sum(ms.pending[ms.n - 1]), len(keys))
            ms = f(ms, keys, keys)
            self.assertEqual(ms.n, 1)
            self.assertEqual(ms.pending[0], MergeRun(0, len(keys)))
            self.assertSorted(orig_keys, keys)