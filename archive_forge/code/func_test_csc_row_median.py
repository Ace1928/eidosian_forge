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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_csc_row_median(csc_container, csr_container):
    rng = np.random.RandomState(0)
    X = rng.rand(100, 50)
    dense_median = np.median(X, axis=0)
    csc = csc_container(X)
    sparse_median = csc_median_axis_0(csc)
    assert_array_equal(sparse_median, dense_median)
    X = rng.rand(51, 100)
    X[X < 0.7] = 0.0
    ind = rng.randint(0, 50, 10)
    X[ind] = -X[ind]
    csc = csc_container(X)
    dense_median = np.median(X, axis=0)
    sparse_median = csc_median_axis_0(csc)
    assert_array_equal(sparse_median, dense_median)
    X = [[0, -2], [-1, -1], [1, 0], [2, 1]]
    csc = csc_container(X)
    assert_array_equal(csc_median_axis_0(csc), np.array([0.5, -0.5]))
    X = [[0, -2], [-1, -5], [1, -3]]
    csc = csc_container(X)
    assert_array_equal(csc_median_axis_0(csc), np.array([0.0, -3]))
    with pytest.raises(TypeError):
        csc_median_axis_0(csr_container(X))