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
def test_1d_multioutput_lasso_and_multitask_lasso_cv():
    X, y, _, _ = build_dataset(n_features=10)
    y = y[:, np.newaxis]
    clf = LassoCV(n_alphas=5, eps=0.002)
    clf.fit(X, y[:, 0])
    clf1 = MultiTaskLassoCV(n_alphas=5, eps=0.002)
    clf1.fit(X, y)
    assert_almost_equal(clf.alpha_, clf1.alpha_)
    assert_almost_equal(clf.coef_, clf1.coef_[0])
    assert_almost_equal(clf.intercept_, clf1.intercept_[0])