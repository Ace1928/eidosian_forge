import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('X, y, expected_f_statistic, expected_p_values, force_finite', [(np.array([[2, 1], [2, 0], [2, 10], [2, 4]]), np.array([0, 1, 1, 0]), np.array([0.0, 0.2293578]), np.array([1.0, 0.67924985]), True), (np.array([[5, 1], [3, 0], [2, 10], [8, 4]]), np.array([0, 0, 0, 0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), True), (np.array([[0, 1], [1, 0], [2, 10], [3, 4]]), np.array([0, 1, 2, 3]), np.array([np.finfo(np.float64).max, 0.845433]), np.array([0.0, 0.454913]), True), (np.array([[3, 1], [2, 0], [1, 10], [0, 4]]), np.array([0, 1, 2, 3]), np.array([np.finfo(np.float64).max, 0.845433]), np.array([0.0, 0.454913]), True), (np.array([[2, 1], [2, 0], [2, 10], [2, 4]]), np.array([0, 1, 1, 0]), np.array([np.nan, 0.2293578]), np.array([np.nan, 0.67924985]), False), (np.array([[5, 1], [3, 0], [2, 10], [8, 4]]), np.array([0, 0, 0, 0]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), False), (np.array([[0, 1], [1, 0], [2, 10], [3, 4]]), np.array([0, 1, 2, 3]), np.array([np.inf, 0.845433]), np.array([0.0, 0.454913]), False), (np.array([[3, 1], [2, 0], [1, 10], [0, 4]]), np.array([0, 1, 2, 3]), np.array([np.inf, 0.845433]), np.array([0.0, 0.454913]), False)])
def test_f_regression_corner_case(X, y, expected_f_statistic, expected_p_values, force_finite):
    """Check the behaviour of `force_finite` for some corner cases with `f_regression`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/15672
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        f_statistic, p_values = f_regression(X, y, force_finite=force_finite)
    np.testing.assert_array_almost_equal(f_statistic, expected_f_statistic)
    np.testing.assert_array_almost_equal(p_values, expected_p_values)