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
def test_adaptive_longer_than_constant(klass):
    clf1 = klass(learning_rate='adaptive', eta0=0.01, tol=0.001, max_iter=100)
    clf1.fit(iris.data, iris.target)
    clf2 = klass(learning_rate='constant', eta0=0.01, tol=0.001, max_iter=100)
    clf2.fit(iris.data, iris.target)
    assert clf1.n_iter_ > clf2.n_iter_