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
def test_enet_path():
    X, y, X_test, y_test = build_dataset(n_samples=200, n_features=100, n_informative_features=100)
    max_iter = 150
    clf = ElasticNetCV(alphas=[0.01, 0.05, 0.1], eps=0.002, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter)
    ignore_warnings(clf.fit)(X, y)
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    assert clf.l1_ratio_ == min(clf.l1_ratio)
    clf = ElasticNetCV(alphas=[0.01, 0.05, 0.1], eps=0.002, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter, precompute=True)
    ignore_warnings(clf.fit)(X, y)
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    assert clf.l1_ratio_ == min(clf.l1_ratio)
    assert clf.score(X_test, y_test) > 0.99
    X, y, X_test, y_test = build_dataset(n_features=10, n_targets=3)
    clf = MultiTaskElasticNetCV(n_alphas=5, eps=0.002, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter)
    ignore_warnings(clf.fit)(X, y)
    assert clf.score(X_test, y_test) > 0.99
    assert clf.coef_.shape == (3, 10)
    X, y, _, _ = build_dataset(n_features=10)
    clf1 = ElasticNetCV(n_alphas=5, eps=0.002, l1_ratio=[0.5, 0.7])
    clf1.fit(X, y)
    clf2 = MultiTaskElasticNetCV(n_alphas=5, eps=0.002, l1_ratio=[0.5, 0.7])
    clf2.fit(X, y[:, np.newaxis])
    assert_almost_equal(clf1.l1_ratio_, clf2.l1_ratio_)
    assert_almost_equal(clf1.alpha_, clf2.alpha_)