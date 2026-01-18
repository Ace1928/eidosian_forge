import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb import _safe_divide
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR
from sklearn.utils import check_random_state, tosequence
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_max_feature_regression(global_random_seed):
    X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=global_random_seed)
    X_train, X_test = (X[:2000], X[2000:])
    y_train, y_test = (y[:2000], y[2000:])
    gbrt = GradientBoostingClassifier(n_estimators=100, min_samples_split=5, max_depth=2, learning_rate=0.1, max_features=2, random_state=global_random_seed)
    gbrt.fit(X_train, y_train)
    log_loss = gbrt._loss(y_test, gbrt.decision_function(X_test))
    assert log_loss < 0.5, 'GB failed with deviance %.4f' % log_loss