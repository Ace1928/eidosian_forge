import warnings
import numpy as np
import pytest
from scipy import linalg, sparse
from sklearn.datasets import load_iris, make_regression, make_sparse_uncorrelated
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import (
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
@pytest.mark.parametrize('fit_intercept', [False, True])
def test_linear_regression_sample_weight_consistency(sparse_container, fit_intercept, global_random_seed):
    """Test that the impact of sample_weight is consistent.

    Note that this test is stricter than the common test
    check_sample_weights_invariance alone and also tests sparse X.
    It is very similar to test_enet_sample_weight_consistency.
    """
    rng = np.random.RandomState(global_random_seed)
    n_samples, n_features = (10, 5)
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    if sparse_container is not None:
        X = sparse_container(X)
    params = dict(fit_intercept=fit_intercept)
    reg = LinearRegression(**params).fit(X, y, sample_weight=None)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight = rng.uniform(low=0.01, high=2, size=X.shape[0])
    reg = reg.fit(X, y, sample_weight=sample_weight)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    reg.fit(X, y, sample_weight=np.pi * sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06 if sparse_container is None else 1e-05)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight_0 = sample_weight.copy()
    sample_weight_0[-5:] = 0
    y[-5:] *= 1000
    reg.fit(X, y, sample_weight=sample_weight_0)
    coef_0 = reg.coef_.copy()
    if fit_intercept:
        intercept_0 = reg.intercept_
    reg.fit(X[:-5], y[:-5], sample_weight=sample_weight[:-5])
    if fit_intercept and sparse_container is None:
        pass
    else:
        assert_allclose(reg.coef_, coef_0, rtol=1e-05)
        if fit_intercept:
            assert_allclose(reg.intercept_, intercept_0)
    if sparse_container is not None:
        X2 = sparse.vstack([X, X[:n_samples // 2]], format='csc')
    else:
        X2 = np.concatenate([X, X[:n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[:n_samples // 2]])
    sample_weight_1 = sample_weight.copy()
    sample_weight_1[:n_samples // 2] *= 2
    sample_weight_2 = np.concatenate([sample_weight, sample_weight[:n_samples // 2]], axis=0)
    reg1 = LinearRegression(**params).fit(X, y, sample_weight=sample_weight_1)
    reg2 = LinearRegression(**params).fit(X2, y2, sample_weight=sample_weight_2)
    assert_allclose(reg1.coef_, reg2.coef_, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg1.intercept_, reg2.intercept_)