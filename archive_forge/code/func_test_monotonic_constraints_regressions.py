import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('TreeRegressor', TREE_BASED_REGRESSOR_CLASSES)
@pytest.mark.parametrize('depth_first_builder', (True, False))
@pytest.mark.parametrize('sparse_splitter', (True, False))
@pytest.mark.parametrize('criterion', ('absolute_error', 'squared_error'))
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_monotonic_constraints_regressions(TreeRegressor, depth_first_builder, sparse_splitter, criterion, global_random_seed, csc_container):
    n_samples = 1000
    n_samples_train = 900
    X, y = make_regression(n_samples=n_samples, n_features=5, n_informative=5, random_state=global_random_seed)
    train = np.arange(n_samples_train)
    test = np.arange(n_samples_train, n_samples)
    X_train = X[train]
    y_train = y[train]
    X_test = np.copy(X[test])
    X_test_incr = np.copy(X_test)
    X_test_decr = np.copy(X_test)
    X_test_incr[:, 0] += 10
    X_test_decr[:, 1] += 10
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    monotonic_cst[1] = -1
    if depth_first_builder:
        est = TreeRegressor(max_depth=None, monotonic_cst=monotonic_cst, criterion=criterion)
    else:
        est = TreeRegressor(max_depth=8, monotonic_cst=monotonic_cst, criterion=criterion, max_leaf_nodes=n_samples_train)
    if hasattr(est, 'random_state'):
        est.set_params(random_state=global_random_seed)
    if hasattr(est, 'n_estimators'):
        est.set_params(**{'n_estimators': 5})
    if sparse_splitter:
        X_train = csc_container(X_train)
    est.fit(X_train, y_train)
    y = est.predict(X_test)
    y_incr = est.predict(X_test_incr)
    assert np.all(y_incr >= y)
    y_decr = est.predict(X_test_decr)
    assert np.all(y_decr <= y)