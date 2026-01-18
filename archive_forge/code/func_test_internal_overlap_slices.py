import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_internal_overlap_slices():
    x = np.zeros([17, 34, 71, 97], dtype=np.int16)
    rng = np.random.RandomState(1234)

    def random_slice(n, step):
        start = rng.randint(0, n + 1, dtype=np.intp)
        stop = rng.randint(start, n + 1, dtype=np.intp)
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = (start, stop)
            step *= -1
        return slice(start, stop, step)
    cases = 0
    min_count = 5000
    while cases < min_count:
        steps = tuple((rng.randint(1, 11, dtype=np.intp) if rng.randint(0, 5, dtype=np.intp) == 0 else 1 for j in range(x.ndim)))
        t1 = np.arange(x.ndim)
        rng.shuffle(t1)
        s1 = tuple((random_slice(p, s) for p, s in zip(x.shape, steps)))
        a = x[s1].transpose(t1)
        assert_(not internal_overlap(a))
        cases += 1