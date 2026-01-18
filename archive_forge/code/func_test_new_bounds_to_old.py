import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
def test_new_bounds_to_old():
    lb = np.array([-np.inf, 2, 3])
    ub = np.array([3, np.inf, 10])
    bounds = [(None, 3), (2, None), (3, 10)]
    assert_array_equal(new_bounds_to_old(lb, ub, 3), bounds)
    bounds_single_lb = [(-1, 3), (-1, None), (-1, 10)]
    assert_array_equal(new_bounds_to_old(-1, ub, 3), bounds_single_lb)
    bounds_no_lb = [(None, 3), (None, None), (None, 10)]
    assert_array_equal(new_bounds_to_old(-np.inf, ub, 3), bounds_no_lb)
    bounds_single_ub = [(None, 20), (2, 20), (3, 20)]
    assert_array_equal(new_bounds_to_old(lb, 20, 3), bounds_single_ub)
    bounds_no_ub = [(None, None), (2, None), (3, None)]
    assert_array_equal(new_bounds_to_old(lb, np.inf, 3), bounds_no_ub)
    bounds_single_both = [(1, 2), (1, 2), (1, 2)]
    assert_array_equal(new_bounds_to_old(1, 2, 3), bounds_single_both)
    bounds_no_both = [(None, None), (None, None), (None, None)]
    assert_array_equal(new_bounds_to_old(-np.inf, np.inf, 3), bounds_no_both)