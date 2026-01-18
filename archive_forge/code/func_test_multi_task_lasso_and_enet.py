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
def test_multi_task_lasso_and_enet():
    X, y, X_test, y_test = build_dataset()
    Y = np.c_[y, y]
    clf = MultiTaskLasso(alpha=1, tol=1e-08).fit(X, Y)
    assert 0 < clf.dual_gap_ < 1e-05
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])
    clf = MultiTaskElasticNet(alpha=1, tol=1e-08).fit(X, Y)
    assert 0 < clf.dual_gap_ < 1e-05
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])
    clf = MultiTaskElasticNet(alpha=1.0, tol=1e-08, max_iter=1)
    warning_message = 'Objective did not converge. You might want to increase the number of iterations.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, Y)