import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('LinearModel, params', [(Lasso, {'tol': 1e-16, 'alpha': 0.1}), (LassoCV, {'tol': 1e-16}), (ElasticNetCV, {}), (RidgeClassifier, {'solver': 'sparse_cg', 'alpha': 0.1}), (ElasticNet, {'tol': 1e-16, 'l1_ratio': 1, 'alpha': 0.01}), (ElasticNet, {'tol': 1e-16, 'l1_ratio': 0, 'alpha': 0.01}), (Ridge, {'solver': 'sparse_cg', 'tol': 1e-12, 'alpha': 0.1}), (LinearRegression, {}), (RidgeCV, {}), (RidgeClassifierCV, {})])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_model_pipeline_same_dense_and_sparse(LinearModel, params, csr_container):
    model_dense = make_pipeline(StandardScaler(with_mean=False), LinearModel(**params))
    model_sparse = make_pipeline(StandardScaler(with_mean=False), LinearModel(**params))
    rng = np.random.RandomState(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0
    X_sparse = csr_container(X)
    y = rng.rand(n_samples)
    if is_classifier(model_dense):
        y = np.sign(y)
    model_dense.fit(X, y)
    model_sparse.fit(X_sparse, y)
    assert_allclose(model_sparse[1].coef_, model_dense[1].coef_)
    y_pred_dense = model_dense.predict(X)
    y_pred_sparse = model_sparse.predict(X_sparse)
    assert_allclose(y_pred_dense, y_pred_sparse)
    assert_allclose(model_dense[1].intercept_, model_sparse[1].intercept_)