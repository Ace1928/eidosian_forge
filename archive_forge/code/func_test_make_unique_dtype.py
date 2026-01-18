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
def test_make_unique_dtype():
    x_list = [2, 2, 2, 3, 5]
    for dtype in (np.float32, np.float64):
        x = np.array(x_list, dtype=dtype)
        y = x.copy()
        w = np.ones_like(x)
        x, y, w = _make_unique(x, y, w)
        assert_array_equal(x, [2, 3, 5])