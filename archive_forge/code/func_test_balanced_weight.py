import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_balanced_weight(klass):
    X, y = (iris.data, iris.target)
    X = scale(X)
    idx = np.arange(X.shape[0])
    rng = np.random.RandomState(6)
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    clf = klass(alpha=0.0001, max_iter=1000, class_weight=None, shuffle=False).fit(X, y)
    f1 = metrics.f1_score(y, clf.predict(X), average='weighted')
    assert_almost_equal(f1, 0.96, decimal=1)
    clf_balanced = klass(alpha=0.0001, max_iter=1000, class_weight='balanced', shuffle=False).fit(X, y)
    f1 = metrics.f1_score(y, clf_balanced.predict(X), average='weighted')
    assert_almost_equal(f1, 0.96, decimal=1)
    assert_array_almost_equal(clf.coef_, clf_balanced.coef_, 6)
    X_0 = X[y == 0, :]
    y_0 = y[y == 0]
    X_imbalanced = np.vstack([X] + [X_0] * 10)
    y_imbalanced = np.concatenate([y] + [y_0] * 10)
    clf = klass(max_iter=1000, class_weight=None, shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average='weighted') < 0.96
    clf = klass(max_iter=1000, class_weight='balanced', shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average='weighted') > 0.96