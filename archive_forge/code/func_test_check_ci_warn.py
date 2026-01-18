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
def test_check_ci_warn():
    x = [0, 1, 2, 3, 4, 5]
    y = [0, -1, 2, -3, 4, -5]
    msg = 'interval'
    with pytest.warns(UserWarning, match=msg):
        is_increasing = check_increasing(x, y)
    assert not is_increasing