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
def test_implicit_center_matmat(global_random_seed, centered_matrices):
    X_sparse_centered, X_dense_centered = centered_matrices
    rng = np.random.default_rng(global_random_seed)
    Y = rng.standard_normal((X_dense_centered.shape[1], 50))
    assert_allclose(X_dense_centered @ Y, X_sparse_centered.matmat(Y))
    assert_allclose(X_dense_centered @ Y, X_sparse_centered @ Y)