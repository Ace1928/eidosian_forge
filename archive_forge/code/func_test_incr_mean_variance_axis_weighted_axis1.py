import numpy as np
import pytest
import scipy.sparse as sp
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import linalg
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.sparsefuncs import (
from sklearn.utils.sparsefuncs_fast import (
@pytest.mark.parametrize(['Xw', 'X', 'weights'], [([[0, 0, 1], [0, 2, 3]], [[0, 0, 1], [0, 2, 3]], [1, 1, 1]), ([[0, 0, 1], [0, 1, 1]], [[0, 0, 0, 1], [0, 1, 1, 1]], [1, 2, 1]), ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1]], None), ([[0, np.nan, 2], [0, np.nan, np.nan]], [[0, np.nan, 2], [0, np.nan, np.nan]], [1.0, 1.0, 1.0]), ([[0, 0], [1, np.nan], [2, 0], [0, 3], [np.nan, np.nan], [np.nan, 2]], [[0, 0, 0], [1, 1, np.nan], [2, 2, 0], [0, 0, 3], [np.nan, np.nan, np.nan], [np.nan, np.nan, 2]], [2.0, 1.0]), ([[1, 0, 1], [0, 3, 1]], [[1, 0, 0, 0, 1], [0, 3, 3, 3, 1]], np.array([1, 3, 1]))])
@pytest.mark.parametrize('sparse_constructor', CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_incr_mean_variance_axis_weighted_axis1(Xw, X, weights, sparse_constructor, dtype):
    axis = 1
    Xw_sparse = sparse_constructor(Xw).astype(dtype)
    X_sparse = sparse_constructor(X).astype(dtype)
    last_mean = np.zeros(np.shape(Xw)[0], dtype=dtype)
    last_var = np.zeros_like(last_mean, dtype=dtype)
    last_n = np.zeros_like(last_mean, dtype=np.int64)
    means0, vars0, n_incr0 = incr_mean_variance_axis(X=X_sparse, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n, weights=None)
    means_w0, vars_w0, n_incr_w0 = incr_mean_variance_axis(X=Xw_sparse, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n, weights=weights)
    assert means_w0.dtype == dtype
    assert vars_w0.dtype == dtype
    assert n_incr_w0.dtype == dtype
    means_simple, vars_simple = mean_variance_axis(X=X_sparse, axis=axis)
    assert_array_almost_equal(means0, means_w0)
    assert_array_almost_equal(means0, means_simple)
    assert_array_almost_equal(vars0, vars_w0)
    assert_array_almost_equal(vars0, vars_simple)
    assert_array_almost_equal(n_incr0, n_incr_w0)
    means1, vars1, n_incr1 = incr_mean_variance_axis(X=X_sparse, axis=axis, last_mean=means0, last_var=vars0, last_n=n_incr0, weights=None)
    means_w1, vars_w1, n_incr_w1 = incr_mean_variance_axis(X=Xw_sparse, axis=axis, last_mean=means_w0, last_var=vars_w0, last_n=n_incr_w0, weights=weights)
    assert_array_almost_equal(means1, means_w1)
    assert_array_almost_equal(vars1, vars_w1)
    assert_array_almost_equal(n_incr1, n_incr_w1)
    assert means_w1.dtype == dtype
    assert vars_w1.dtype == dtype
    assert n_incr_w1.dtype == dtype