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
@pytest.mark.parametrize('lil_container', LIL_CONTAINERS)
def test_mean_variance_axis1(csc_container, csr_container, lil_container):
    X, _ = make_classification(5, 4, random_state=0)
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_lil = lil_container(X)
    X_lil[1, 0] = 0
    X[1, 0] = 0
    with pytest.raises(TypeError):
        mean_variance_axis(X_lil, axis=1)
    X_csr = csr_container(X_lil)
    X_csc = csc_container(X_lil)
    expected_dtypes = [(np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64), (np.int64, np.float64)]
    for input_dtype, output_dtype in expected_dtypes:
        X_test = X.astype(input_dtype)
        for X_sparse in (X_csr, X_csc):
            X_sparse = X_sparse.astype(input_dtype)
            X_means, X_vars = mean_variance_axis(X_sparse, axis=0)
            assert X_means.dtype == output_dtype
            assert X_vars.dtype == output_dtype
            assert_array_almost_equal(X_means, np.mean(X_test, axis=0))
            assert_array_almost_equal(X_vars, np.var(X_test, axis=0))