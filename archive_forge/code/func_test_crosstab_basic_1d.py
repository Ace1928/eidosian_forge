import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.stats.contingency import crosstab
def test_crosstab_basic_1d():
    x = [1, 2, 3, 1, 2, 3, 3]
    expected_xvals = [1, 2, 3]
    expected_count = np.array([2, 2, 3])
    (xvals,), count = crosstab(x)
    assert_array_equal(xvals, expected_xvals)
    assert_array_equal(count, expected_count)