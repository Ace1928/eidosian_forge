import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('TreeRegressor', TREE_REGRESSOR_CLASSES)
@pytest.mark.parametrize('monotonic_sign', (-1, 1))
@pytest.mark.parametrize('depth_first_builder', (True, False))
@pytest.mark.parametrize('criterion', ('absolute_error', 'squared_error'))
def test_1d_tree_nodes_values(TreeRegressor, monotonic_sign, depth_first_builder, criterion, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    n_features = 1
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    if depth_first_builder:
        clf = TreeRegressor(monotonic_cst=[monotonic_sign], criterion=criterion, random_state=global_random_seed)
    else:
        clf = TreeRegressor(monotonic_cst=[monotonic_sign], max_leaf_nodes=n_samples, criterion=criterion, random_state=global_random_seed)
    clf.fit(X, y)
    assert_1d_reg_tree_children_monotonic_bounded(clf.tree_, monotonic_sign)
    assert_1d_reg_monotonic(clf, monotonic_sign, np.min(X), np.max(X), 100)