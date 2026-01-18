import os
import re
import sys
from contextlib import contextmanager
import numpy as np
import pytest
from numpy.testing import (
from scipy.linalg import norm
from scipy.optimize import fmin_bfgs
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model._theil_sen import (
from sklearn.utils._testing import assert_almost_equal
def test_less_samples_than_features():
    random_state = np.random.RandomState(0)
    n_samples, n_features = (10, 20)
    X = random_state.normal(size=(n_samples, n_features))
    y = random_state.normal(size=n_samples)
    theil_sen = TheilSenRegressor(fit_intercept=False, random_state=0).fit(X, y)
    lstq = LinearRegression(fit_intercept=False).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, lstq.coef_, 12)
    theil_sen = TheilSenRegressor(fit_intercept=True, random_state=0).fit(X, y)
    y_pred = theil_sen.predict(X)
    assert_array_almost_equal(y_pred, y, 12)