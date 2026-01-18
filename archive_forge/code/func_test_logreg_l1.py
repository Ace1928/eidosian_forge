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
def test_logreg_l1():
    rng = np.random.RandomState(42)
    n_samples = 50
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=0)
    X_noise = rng.normal(size=(n_samples, 3))
    X_constant = np.ones(shape=(n_samples, 2))
    X = np.concatenate((X, X_noise, X_constant), axis=1)
    lr_liblinear = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', fit_intercept=False, multi_class='ovr', tol=1e-10)
    lr_liblinear.fit(X, y)
    lr_saga = LogisticRegression(penalty='l1', C=1.0, solver='saga', fit_intercept=False, multi_class='ovr', max_iter=1000, tol=1e-10)
    lr_saga.fit(X, y)
    assert_array_almost_equal(lr_saga.coef_, lr_liblinear.coef_)
    assert_array_almost_equal(lr_liblinear.coef_[0, -5:], np.zeros(5))
    assert_array_almost_equal(lr_saga.coef_[0, -5:], np.zeros(5))