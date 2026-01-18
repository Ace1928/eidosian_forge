import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import (
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.validation import check_array
def test_fast_predict():
    rng = np.random.RandomState(123)
    n_samples = 10 ** 3
    X_train = 20.0 * rng.rand(n_samples) - 10
    y_train = np.less(rng.rand(n_samples), expit(X_train)).astype('int64').astype('float64')
    weights = rng.rand(n_samples)
    weights[rng.rand(n_samples) < 0.1] = 0
    slow_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    fast_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    X_train_fit, y_train_fit = slow_model._build_y(X_train, y_train, sample_weight=weights, trim_duplicates=False)
    slow_model._build_f(X_train_fit, y_train_fit)
    fast_model.fit(X_train, y_train, sample_weight=weights)
    X_test = 20.0 * rng.rand(n_samples) - 10
    y_pred_slow = slow_model.predict(X_test)
    y_pred_fast = fast_model.predict(X_test)
    assert_array_equal(y_pred_slow, y_pred_fast)