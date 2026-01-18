import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_mutual_info_classif_discrete(global_dtype):
    X = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype)
    y = np.array([0, 1, 2, 2, 1])
    mi = mutual_info_classif(X, y, discrete_features=True)
    assert_array_equal(np.argsort(-mi), np.array([0, 2, 1]))