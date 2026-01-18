import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
def test_min_max_scaler_zero_variance_features():
    X = [[0.0, 1.0, +0.5], [0.0, 1.0, -0.1], [0.0, 1.0, +1.1]]
    X_new = [[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]]
    scaler = MinMaxScaler()
    X_trans = scaler.fit_transform(X)
    X_expected_0_1 = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert_array_almost_equal(X_trans, X_expected_0_1)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    X_trans_new = scaler.transform(X_new)
    X_expected_0_1_new = [[+0.0, 1.0, 0.5], [-1.0, 0.0, 0.083], [+0.0, 0.0, 1.333]]
    assert_array_almost_equal(X_trans_new, X_expected_0_1_new, decimal=2)
    scaler = MinMaxScaler(feature_range=(1, 2))
    X_trans = scaler.fit_transform(X)
    X_expected_1_2 = [[1.0, 1.0, 1.5], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]]
    assert_array_almost_equal(X_trans, X_expected_1_2)
    X_trans = minmax_scale(X)
    assert_array_almost_equal(X_trans, X_expected_0_1)
    X_trans = minmax_scale(X, feature_range=(1, 2))
    assert_array_almost_equal(X_trans, X_expected_1_2)