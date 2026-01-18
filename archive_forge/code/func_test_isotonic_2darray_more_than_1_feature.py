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
def test_isotonic_2darray_more_than_1_feature():
    X = np.arange(10)
    X_2d = np.c_[X, X]
    y = np.arange(10)
    msg = 'should be a 1d array or 2d array with 1 feature'
    with pytest.raises(ValueError, match=msg):
        IsotonicRegression().fit(X_2d, y)
    iso_reg = IsotonicRegression().fit(X, y)
    with pytest.raises(ValueError, match=msg):
        iso_reg.predict(X_2d)
    with pytest.raises(ValueError, match=msg):
        iso_reg.transform(X_2d)