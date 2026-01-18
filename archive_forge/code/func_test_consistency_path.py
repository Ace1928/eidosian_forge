import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def test_consistency_path():
    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = [1] * 100 + [-1] * 100
    Cs = np.logspace(0, 4, 10)
    f = ignore_warnings
    for solver in ['sag', 'saga']:
        coefs, Cs, _ = f(_logistic_regression_path)(X, y, Cs=Cs, fit_intercept=False, tol=1e-05, solver=solver, max_iter=1000, multi_class='ovr', random_state=0)
        for i, C in enumerate(Cs):
            lr = LogisticRegression(C=C, fit_intercept=False, tol=1e-05, solver=solver, multi_class='ovr', random_state=0, max_iter=1000)
            lr.fit(X, y)
            lr_coef = lr.coef_.ravel()
            assert_array_almost_equal(lr_coef, coefs[i], decimal=4, err_msg='with solver = %s' % solver)
    for solver in ('lbfgs', 'newton-cg', 'newton-cholesky', 'liblinear', 'sag', 'saga'):
        Cs = [1000.0]
        coefs, Cs, _ = f(_logistic_regression_path)(X, y, Cs=Cs, tol=1e-06, solver=solver, intercept_scaling=10000.0, random_state=0, multi_class='ovr')
        lr = LogisticRegression(C=Cs[0], tol=1e-06, intercept_scaling=10000.0, random_state=0, multi_class='ovr', solver=solver)
        lr.fit(X, y)
        lr_coef = np.concatenate([lr.coef_.ravel(), lr.intercept_])
        assert_array_almost_equal(lr_coef, coefs[0], decimal=4, err_msg='with solver = %s' % solver)