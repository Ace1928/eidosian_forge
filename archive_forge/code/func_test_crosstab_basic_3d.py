import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.stats.contingency import crosstab
def test_crosstab_basic_3d():
    a = 'a'
    b = 'b'
    x = [0, 0, 9, 9, 0, 0, 9, 9]
    y = [a, a, a, a, b, b, b, a]
    z = [1, 2, 3, 1, 2, 3, 3, 1]
    expected_xvals = [0, 9]
    expected_yvals = [a, b]
    expected_zvals = [1, 2, 3]
    expected_count = np.array([[[1, 1, 0], [0, 1, 1]], [[2, 0, 1], [0, 0, 1]]])
    (xvals, yvals, zvals), count = crosstab(x, y, z)
    assert_array_equal(xvals, expected_xvals)
    assert_array_equal(yvals, expected_yvals)
    assert_array_equal(zvals, expected_zvals)
    assert_array_equal(count, expected_count)