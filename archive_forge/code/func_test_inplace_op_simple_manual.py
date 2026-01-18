import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_inplace_op_simple_manual(self):
    rng = np.random.RandomState(1234)
    x = rng.rand(200, 200)
    x += x.T
    assert_array_equal(x - x.T, 0)