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
def test_gradient_boosting_validation_fraction():
    X, y = make_classification(n_samples=1000, random_state=0)
    gbc = GradientBoostingClassifier(n_estimators=100, n_iter_no_change=10, validation_fraction=0.1, learning_rate=0.1, max_depth=3, random_state=42)
    gbc2 = clone(gbc).set_params(validation_fraction=0.3)
    gbc3 = clone(gbc).set_params(n_iter_no_change=20)
    gbr = GradientBoostingRegressor(n_estimators=100, n_iter_no_change=10, learning_rate=0.1, max_depth=3, validation_fraction=0.1, random_state=42)
    gbr2 = clone(gbr).set_params(validation_fraction=0.3)
    gbr3 = clone(gbr).set_params(n_iter_no_change=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    gbc.fit(X_train, y_train)
    gbc2.fit(X_train, y_train)
    assert gbc.n_estimators_ != gbc2.n_estimators_
    gbr.fit(X_train, y_train)
    gbr2.fit(X_train, y_train)
    assert gbr.n_estimators_ != gbr2.n_estimators_
    gbc3.fit(X_train, y_train)
    gbr3.fit(X_train, y_train)
    assert gbr.n_estimators_ < gbr3.n_estimators_
    assert gbc.n_estimators_ < gbc3.n_estimators_