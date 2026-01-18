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
def test_check_increasing_down_extreme():
    x = [0, 1, 2, 3, 4, 5]
    y = [0, -1, -2, -3, -4, -5]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        is_increasing = check_increasing(x, y)
    assert not is_increasing