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
def test_isotonic_duplicate_min_entry():
    x = [0, 0, 1]
    y = [0, 0, 1]
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(x, y)
    all_predictions_finite = np.all(np.isfinite(ir.predict(x)))
    assert all_predictions_finite