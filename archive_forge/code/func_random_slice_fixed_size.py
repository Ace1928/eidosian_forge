import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def random_slice_fixed_size(n, step, size):
    start = rng.randint(0, n + 1 - size * step)
    stop = start + (size - 1) * step + 1
    if rng.randint(0, 2) == 0:
        stop, start = (start - 1, stop - 1)
        if stop < 0:
            stop = None
        step *= -1
    return slice(start, stop, step)