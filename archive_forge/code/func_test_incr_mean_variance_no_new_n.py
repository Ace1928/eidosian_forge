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
def test_incr_mean_variance_no_new_n():
    axis = 0
    X1 = sp.random(5, 1, density=0.8, random_state=0).tocsr()
    X2 = sp.random(0, 1, density=0.8, random_state=0).tocsr()
    last_mean, last_var = (np.zeros(X1.shape[1]), np.zeros(X1.shape[1]))
    last_n = np.zeros(X1.shape[1], dtype=np.int64)
    last_mean, last_var, last_n = incr_mean_variance_axis(X1, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n)
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(X2, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n)
    assert_allclose(updated_mean, last_mean)
    assert_allclose(updated_var, last_var)
    assert_allclose(updated_n, last_n)