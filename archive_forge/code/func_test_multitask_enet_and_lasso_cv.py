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
def test_multitask_enet_and_lasso_cv():
    X, y, _, _ = build_dataset(n_features=50, n_targets=3)
    clf = MultiTaskElasticNetCV(cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.00556, 3)
    clf = MultiTaskLassoCV(cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.00278, 3)
    X, y, _, _ = build_dataset(n_targets=3)
    clf = MultiTaskElasticNetCV(n_alphas=10, eps=0.001, max_iter=100, l1_ratio=[0.3, 0.5], tol=0.001, cv=3)
    clf.fit(X, y)
    assert 0.5 == clf.l1_ratio_
    assert (3, X.shape[1]) == clf.coef_.shape
    assert (3,) == clf.intercept_.shape
    assert (2, 10, 3) == clf.mse_path_.shape
    assert (2, 10) == clf.alphas_.shape
    X, y, _, _ = build_dataset(n_targets=3)
    clf = MultiTaskLassoCV(n_alphas=10, eps=0.001, max_iter=100, tol=0.001, cv=3)
    clf.fit(X, y)
    assert (3, X.shape[1]) == clf.coef_.shape
    assert (3,) == clf.intercept_.shape
    assert (10, 3) == clf.mse_path_.shape
    assert 10 == len(clf.alphas_)