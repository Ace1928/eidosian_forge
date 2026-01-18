import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_mutual_info_regression(global_dtype):
    T = np.array([[1, 0.5, 2, 1], [0, 1, 0.1, 0.0], [0, 0.1, 1, 0.1], [0, 0.1, 0.1, 1]])
    cov = T.dot(T.T)
    mean = np.zeros(4)
    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(global_dtype, copy=False)
    X = Z[:, 1:]
    y = Z[:, 0]
    mi = mutual_info_regression(X, y, random_state=0)
    assert_array_equal(np.argsort(-mi), np.array([1, 2, 0]))
    assert mi.dtype == np.float64