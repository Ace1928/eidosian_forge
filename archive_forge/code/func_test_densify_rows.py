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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_densify_rows(csr_container):
    for dtype in (np.float32, np.float64):
        X = csr_container([[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=dtype)
        X_rows = np.array([0, 2, 3], dtype=np.intp)
        out = np.ones((6, X.shape[1]), dtype=dtype)
        out_rows = np.array([1, 3, 4], dtype=np.intp)
        expect = np.ones_like(out)
        expect[out_rows] = X[X_rows, :].toarray()
        assign_rows_csr(X, X_rows, out_rows, out)
        assert_array_equal(out, expect)