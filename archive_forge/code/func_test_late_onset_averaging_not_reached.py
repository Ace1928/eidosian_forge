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
@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor, SGDOneClassSVM, SparseSGDOneClassSVM])
def test_late_onset_averaging_not_reached(klass):
    clf1 = klass(average=600)
    clf2 = klass()
    for _ in range(100):
        if is_classifier(clf1):
            clf1.partial_fit(X, Y, classes=np.unique(Y))
            clf2.partial_fit(X, Y, classes=np.unique(Y))
        else:
            clf1.partial_fit(X, Y)
            clf2.partial_fit(X, Y)
    assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=16)
    if klass in [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]:
        assert_almost_equal(clf1.intercept_, clf2.intercept_, decimal=16)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        assert_allclose(clf1.offset_, clf2.offset_)