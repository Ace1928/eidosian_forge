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
def test_isotonic_regression_oob_raise():
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    x = np.arange(len(y))
    ir = IsotonicRegression(increasing='auto', out_of_bounds='raise')
    ir.fit(x, y)
    msg = 'in x_new is below the interpolation range'
    with pytest.raises(ValueError, match=msg):
        ir.predict([min(x) - 10, max(x) + 10])