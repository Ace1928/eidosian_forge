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
def test_incr_mean_variance_n_float():
    axis = 0
    X = sp.random(5, 2, density=0.8, random_state=0).tocsr()
    last_mean, last_var = (np.zeros(X.shape[1]), np.zeros(X.shape[1]))
    last_n = 0
    _, _, new_n = incr_mean_variance_axis(X, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n)
    assert_allclose(new_n, np.full(X.shape[1], X.shape[0]))