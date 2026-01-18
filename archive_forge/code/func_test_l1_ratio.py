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
def test_l1_ratio():
    X, y = datasets.make_classification(n_samples=1000, n_features=100, n_informative=20, random_state=1234)
    est_en = SGDClassifier(alpha=0.001, penalty='elasticnet', tol=None, max_iter=6, l1_ratio=0.9999999999, random_state=42).fit(X, y)
    est_l1 = SGDClassifier(alpha=0.001, penalty='l1', max_iter=6, random_state=42, tol=None).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l1.coef_)
    est_en = SGDClassifier(alpha=0.001, penalty='elasticnet', tol=None, max_iter=6, l1_ratio=1e-10, random_state=42).fit(X, y)
    est_l2 = SGDClassifier(alpha=0.001, penalty='l2', max_iter=6, random_state=42, tol=None).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l2.coef_)