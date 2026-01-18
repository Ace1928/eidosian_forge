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
@pytest.mark.parametrize('GradientBoostingEstimator', [GradientBoostingClassifier, GradientBoostingRegressor])
def test_gradient_boosting_early_stopping(GradientBoostingEstimator):
    X, y = make_classification(n_samples=1000, random_state=0)
    n_estimators = 1000
    gb_large_tol = GradientBoostingEstimator(n_estimators=n_estimators, n_iter_no_change=10, learning_rate=0.1, max_depth=3, random_state=42, tol=0.1)
    gb_small_tol = GradientBoostingEstimator(n_estimators=n_estimators, n_iter_no_change=10, learning_rate=0.1, max_depth=3, random_state=42, tol=0.001)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    gb_large_tol.fit(X_train, y_train)
    gb_small_tol.fit(X_train, y_train)
    assert gb_large_tol.n_estimators_ < gb_small_tol.n_estimators_ < n_estimators
    assert gb_large_tol.score(X_test, y_test) > 0.7
    assert gb_small_tol.score(X_test, y_test) > 0.7