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
@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_late_onset_averaging_reached(klass):
    eta0 = 0.001
    alpha = 0.0001
    Y_encode = np.array(Y)
    Y_encode[Y_encode == 1] = -1.0
    Y_encode[Y_encode == 2] = 1.0
    clf1 = klass(average=7, learning_rate='constant', loss='squared_error', eta0=eta0, alpha=alpha, max_iter=2, shuffle=False)
    clf2 = klass(average=0, learning_rate='constant', loss='squared_error', eta0=eta0, alpha=alpha, max_iter=1, shuffle=False)
    clf1.fit(X, Y_encode)
    clf2.fit(X, Y_encode)
    average_weights, average_intercept = asgd(klass, X, Y_encode, eta0, alpha, weight_init=clf2.coef_.ravel(), intercept_init=clf2.intercept_)
    assert_array_almost_equal(clf1.coef_.ravel(), average_weights.ravel(), decimal=16)
    assert_almost_equal(clf1.intercept_, average_intercept, decimal=16)