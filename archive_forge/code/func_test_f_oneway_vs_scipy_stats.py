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
def test_f_oneway_vs_scipy_stats():
    rng = np.random.RandomState(0)
    X1 = rng.randn(10, 3)
    X2 = 1 + rng.randn(10, 3)
    f, pv = stats.f_oneway(X1, X2)
    f2, pv2 = f_oneway(X1, X2)
    assert np.allclose(f, f2)
    assert np.allclose(pv, pv2)