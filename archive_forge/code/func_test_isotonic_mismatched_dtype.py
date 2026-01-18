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
@pytest.mark.parametrize('y_dtype', [np.int32, np.int64, np.float32, np.float64])
def test_isotonic_mismatched_dtype(y_dtype):
    reg = IsotonicRegression()
    y = np.array([2, 1, 4, 3, 5], dtype=y_dtype)
    X = np.arange(len(y), dtype=np.float32)
    reg.fit(X, y)
    assert reg.predict(X).dtype == X.dtype