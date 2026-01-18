import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_array_almost_equal_matrix():
    m1 = np.matrix([[1.0, 2.0]])
    m2 = np.matrix([[1.0, np.nan]])
    m3 = np.matrix([[1.0, -np.inf]])
    m4 = np.matrix([[np.nan, np.inf]])
    m5 = np.matrix([[1.0, 2.0], [np.nan, np.inf]])
    for assert_func in (assert_array_almost_equal, assert_almost_equal):
        for m in (m1, m2, m3, m4, m5):
            assert_func(m, m)
            a = np.array(m)
            assert_func(a, m)
            assert_func(m, a)