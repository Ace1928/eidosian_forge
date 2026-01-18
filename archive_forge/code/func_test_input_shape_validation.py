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
def test_input_shape_validation():
    X = np.arange(10)
    X_2d = X.reshape(-1, 1)
    y = np.arange(10)
    iso_reg = IsotonicRegression().fit(X, y)
    iso_reg_2d = IsotonicRegression().fit(X_2d, y)
    assert iso_reg.X_max_ == iso_reg_2d.X_max_
    assert iso_reg.X_min_ == iso_reg_2d.X_min_
    assert iso_reg.y_max == iso_reg_2d.y_max
    assert iso_reg.y_min == iso_reg_2d.y_min
    assert_array_equal(iso_reg.X_thresholds_, iso_reg_2d.X_thresholds_)
    assert_array_equal(iso_reg.y_thresholds_, iso_reg_2d.y_thresholds_)
    y_pred1 = iso_reg.predict(X)
    y_pred2 = iso_reg_2d.predict(X_2d)
    assert_allclose(y_pred1, y_pred2)