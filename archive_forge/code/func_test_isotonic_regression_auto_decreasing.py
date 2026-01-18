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
def test_isotonic_regression_auto_decreasing():
    y = np.array([10, 9, 10, 7, 6, 6.1, 5])
    x = np.arange(len(y))
    ir = IsotonicRegression(increasing='auto')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        y_ = ir.fit_transform(x, y)
        assert all(['invalid value encountered in ' in str(warn.message) for warn in w])
    is_increasing = y_[0] < y_[-1]
    assert not is_increasing