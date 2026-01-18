import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_f_regression_center():
    X = np.arange(-5, 6).reshape(-1, 1)
    n_samples = X.size
    Y = np.ones(n_samples)
    Y[::2] *= -1.0
    Y[0] = 0.0
    F1, _ = f_regression(X, Y, center=True)
    F2, _ = f_regression(X, Y, center=False)
    assert_allclose(F1 * (n_samples - 1.0) / (n_samples - 2.0), F2)
    assert_almost_equal(F2[0], 0.232558139)