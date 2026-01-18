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
def test_lasso_cv():
    X, y, X_test, y_test = build_dataset()
    max_iter = 150
    clf = LassoCV(n_alphas=10, eps=0.001, max_iter=max_iter, cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.056, 2)
    clf = LassoCV(n_alphas=10, eps=0.001, max_iter=max_iter, precompute=True, cv=3)
    clf.fit(X, y)
    assert_almost_equal(clf.alpha_, 0.056, 2)
    lars = LassoLarsCV(max_iter=30, cv=3).fit(X, y)
    assert np.abs(np.searchsorted(clf.alphas_[::-1], lars.alpha_) - np.searchsorted(clf.alphas_[::-1], clf.alpha_)) <= 1
    mse_lars = interpolate.interp1d(lars.cv_alphas_, lars.mse_path_.T)
    np.testing.assert_approx_equal(mse_lars(clf.alphas_[5]).mean(), clf.mse_path_[5].mean(), significant=2)
    assert clf.score(X_test, y_test) > 0.99