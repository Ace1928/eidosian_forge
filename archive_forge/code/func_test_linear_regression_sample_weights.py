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
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_linear_regression_sample_weights(sparse_container, fit_intercept, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples, n_features = (6, 5)
    X = rng.normal(size=(n_samples, n_features))
    if sparse_container is not None:
        X = sparse_container(X)
    y = rng.normal(size=n_samples)
    sample_weight = 1.0 + rng.uniform(size=n_samples)
    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X, y, sample_weight=sample_weight)
    coefs1 = reg.coef_
    inter1 = reg.intercept_
    assert reg.coef_.shape == (X.shape[1],)
    W = np.diag(sample_weight)
    X_aug = X if not fit_intercept else add_dummy_feature(X)
    Xw = X_aug.T @ W @ X_aug
    yw = X_aug.T @ W @ y
    coefs2 = linalg.solve(Xw, yw)
    if not fit_intercept:
        assert_allclose(coefs1, coefs2)
    else:
        assert_allclose(coefs1, coefs2[1:])
        assert_allclose(inter1, coefs2[0])