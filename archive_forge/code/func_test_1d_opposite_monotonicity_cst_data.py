import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('TreeRegressor', TREE_REGRESSOR_CLASSES)
def test_1d_opposite_monotonicity_cst_data(TreeRegressor):
    X = np.linspace(-2, 2, 10).reshape(-1, 1)
    y = X.ravel()
    clf = TreeRegressor(monotonic_cst=[-1])
    clf.fit(X, y)
    assert clf.tree_.node_count == 1
    assert clf.tree_.value[0] == 0.0
    clf = TreeRegressor(monotonic_cst=[1])
    clf.fit(X, -y)
    assert clf.tree_.node_count == 1
    assert clf.tree_.value[0] == 0.0