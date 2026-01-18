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
@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_late_onset_averaging_reached_oneclass(klass):
    eta0 = 0.001
    nu = 0.05
    clf1 = klass(average=7, learning_rate='constant', eta0=eta0, nu=nu, max_iter=2, shuffle=False)
    clf2 = klass(average=0, learning_rate='constant', eta0=eta0, nu=nu, max_iter=1, shuffle=False)
    clf1.fit(X)
    clf2.fit(X)
    average_coef, average_offset = asgd_oneclass(klass, X, eta0, nu, coef_init=clf2.coef_.ravel(), offset_init=clf2.offset_)
    assert_allclose(clf1.coef_.ravel(), average_coef.ravel())
    assert_allclose(clf1.offset_, average_offset)