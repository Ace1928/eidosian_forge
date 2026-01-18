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
def test_lasso_dual_gap():
    """
    Check that Lasso.dual_gap_ matches its objective formulation, with the
    datafit normalized by n_samples
    """
    X, y, _, _ = build_dataset(n_samples=10, n_features=30)
    n_samples = len(y)
    alpha = 0.01 * np.max(np.abs(X.T @ y)) / n_samples
    clf = Lasso(alpha=alpha, fit_intercept=False).fit(X, y)
    w = clf.coef_
    R = y - X @ w
    primal = 0.5 * np.mean(R ** 2) + clf.alpha * np.sum(np.abs(w))
    R /= np.max(np.abs(X.T @ R) / (n_samples * alpha))
    dual = 0.5 * (np.mean(y ** 2) - np.mean((y - R) ** 2))
    assert_allclose(clf.dual_gap_, primal - dual)