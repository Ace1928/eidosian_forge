import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_apply_along_axis_matrix():

    def double(row):
        return row * 2
    m = np.matrix([[0, 1], [2, 3]])
    expected = np.matrix([[0, 2], [4, 6]])
    result = np.apply_along_axis(double, 0, m)
    assert_(isinstance(result, np.matrix))
    assert_array_equal(result, expected)
    result = np.apply_along_axis(double, 1, m)
    assert_(isinstance(result, np.matrix))
    assert_array_equal(result, expected)