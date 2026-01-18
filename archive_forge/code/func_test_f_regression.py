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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_f_regression(csr_container):
    X, y = make_regression(n_samples=200, n_features=20, n_informative=5, shuffle=False, random_state=0)
    F, pv = f_regression(X, y)
    assert (F > 0).all()
    assert (pv > 0).all()
    assert (pv < 1).all()
    assert (pv[:5] < 0.05).all()
    assert (pv[5:] > 0.0001).all()
    F, pv = f_regression(X, y, center=True)
    F_sparse, pv_sparse = f_regression(csr_container(X), y, center=True)
    assert_allclose(F_sparse, F)
    assert_allclose(pv_sparse, pv)
    F, pv = f_regression(X, y, center=False)
    F_sparse, pv_sparse = f_regression(csr_container(X), y, center=False)
    assert_allclose(F_sparse, F)
    assert_allclose(pv_sparse, pv)