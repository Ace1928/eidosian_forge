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
def test_isotonic_non_regression_inf_slope():
    X = np.array([0.0, 4.1e-320, 4.4e-314, 1.0])
    y = np.array([0.42, 0.42, 0.44, 0.44])
    ireg = IsotonicRegression().fit(X, y)
    y_pred = ireg.predict(np.array([0, 2.1e-319, 5.4e-316, 1e-10]))
    assert np.all(np.isfinite(y_pred))