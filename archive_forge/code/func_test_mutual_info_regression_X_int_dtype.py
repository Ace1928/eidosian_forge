import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_mutual_info_regression_X_int_dtype(global_random_seed):
    """Check that results agree when X is integer dtype and float dtype.

    Non-regression test for Issue #26696.
    """
    rng = np.random.RandomState(global_random_seed)
    X = rng.randint(100, size=(100, 10))
    X_float = X.astype(np.float64, copy=True)
    y = rng.randint(100, size=100)
    expected = mutual_info_regression(X_float, y, random_state=global_random_seed)
    result = mutual_info_regression(X, y, random_state=global_random_seed)
    assert_allclose(result, expected)