import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_may_share_memory_bad_max_work():
    x = np.zeros([1])
    assert_raises(OverflowError, np.may_share_memory, x, x, max_work=10 ** 100)
    assert_raises(OverflowError, np.shares_memory, x, x, max_work=10 ** 100)