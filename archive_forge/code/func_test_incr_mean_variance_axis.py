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
def test_incr_mean_variance_axis(csc_container, csr_container, lil_container):
    for axis in [0, 1]:
        rng = np.random.RandomState(0)
        n_features = 50
        n_samples = 10
        if axis == 0:
            data_chunks = [rng.randint(0, 2, size=n_features) for i in range(n_samples)]
        else:
            data_chunks = [rng.randint(0, 2, size=n_samples) for i in range(n_features)]
        last_mean = np.zeros(n_features) if axis == 0 else np.zeros(n_samples)
        last_var = np.zeros_like(last_mean)
        last_n = np.zeros_like(last_mean, dtype=np.int64)
        X = np.array(data_chunks[0])
        X = np.atleast_2d(X)
        X = X.T if axis == 1 else X
        X_lil = lil_container(X)
        X_csr = csr_container(X_lil)
        with pytest.raises(TypeError):
            incr_mean_variance_axis(X=axis, axis=last_mean, last_mean=last_var, last_var=last_n)
        with pytest.raises(TypeError):
            incr_mean_variance_axis(X_lil, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n)
        X_means, X_vars = mean_variance_axis(X_csr, axis)
        X_means_incr, X_vars_incr, n_incr = incr_mean_variance_axis(X_csr, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n)
        assert_array_almost_equal(X_means, X_means_incr)
        assert_array_almost_equal(X_vars, X_vars_incr)
        assert_array_equal(X.shape[axis], n_incr)
        X_csc = csc_container(X_lil)
        X_means, X_vars = mean_variance_axis(X_csc, axis)
        assert_array_almost_equal(X_means, X_means_incr)
        assert_array_almost_equal(X_vars, X_vars_incr)
        assert_array_equal(X.shape[axis], n_incr)
        X = np.vstack(data_chunks)
        X = X.T if axis == 1 else X
        X_lil = lil_container(X)
        X_csr = csr_container(X_lil)
        X_csc = csc_container(X_lil)
        expected_dtypes = [(np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64), (np.int64, np.float64)]
        for input_dtype, output_dtype in expected_dtypes:
            for X_sparse in (X_csr, X_csc):
                X_sparse = X_sparse.astype(input_dtype)
                last_mean = last_mean.astype(output_dtype)
                last_var = last_var.astype(output_dtype)
                X_means, X_vars = mean_variance_axis(X_sparse, axis)
                X_means_incr, X_vars_incr, n_incr = incr_mean_variance_axis(X_sparse, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n)
                assert X_means_incr.dtype == output_dtype
                assert X_vars_incr.dtype == output_dtype
                assert_array_almost_equal(X_means, X_means_incr)
                assert_array_almost_equal(X_vars, X_vars_incr)
                assert_array_equal(X.shape[axis], n_incr)