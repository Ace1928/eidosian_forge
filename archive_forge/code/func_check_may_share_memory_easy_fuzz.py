import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def check_may_share_memory_easy_fuzz(get_max_work, same_steps, min_count):
    x = np.zeros([17, 34, 71, 97], dtype=np.int16)
    feasible = 0
    infeasible = 0
    pair_iter = iter_random_view_pairs(x, same_steps)
    while min(feasible, infeasible) < min_count:
        a, b = next(pair_iter)
        bounds_overlap = np.may_share_memory(a, b)
        may_share_answer = np.may_share_memory(a, b)
        easy_answer = np.may_share_memory(a, b, max_work=get_max_work(a, b))
        exact_answer = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)
        if easy_answer != exact_answer:
            assert_equal(easy_answer, exact_answer)
        if may_share_answer != bounds_overlap:
            assert_equal(may_share_answer, bounds_overlap)
        if bounds_overlap:
            if exact_answer:
                feasible += 1
            else:
                infeasible += 1