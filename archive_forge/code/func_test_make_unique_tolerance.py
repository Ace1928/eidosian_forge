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
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_make_unique_tolerance(dtype):
    x = np.array([0, 1e-16, 1, 1 + 1e-14], dtype=dtype)
    y = x.copy()
    w = np.ones_like(x)
    x, y, w = _make_unique(x, y, w)
    if dtype == np.float64:
        x_out = np.array([0, 1, 1 + 1e-14])
    else:
        x_out = np.array([0, 1])
    assert_array_equal(x, x_out)