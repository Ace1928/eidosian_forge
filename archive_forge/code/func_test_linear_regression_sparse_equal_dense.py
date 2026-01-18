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
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_linear_regression_sparse_equal_dense(fit_intercept, csr_container):
    rng = np.random.RandomState(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0
    Xcsr = csr_container(X)
    y = rng.rand(n_samples)
    params = dict(fit_intercept=fit_intercept)
    clf_dense = LinearRegression(**params)
    clf_sparse = LinearRegression(**params)
    clf_dense.fit(X, y)
    clf_sparse.fit(Xcsr, y)
    assert clf_dense.intercept_ == pytest.approx(clf_sparse.intercept_)
    assert_allclose(clf_dense.coef_, clf_sparse.coef_)