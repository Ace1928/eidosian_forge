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
def test_f_regression_input_dtype():
    rng = np.random.RandomState(0)
    X = rng.rand(10, 20)
    y = np.arange(10).astype(int)
    F1, pv1 = f_regression(X, y)
    F2, pv2 = f_regression(X, y.astype(float))
    assert_allclose(F1, F2, 5)
    assert_allclose(pv1, pv2, 5)