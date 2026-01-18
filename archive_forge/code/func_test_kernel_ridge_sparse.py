import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._testing import assert_array_almost_equal, ignore_warnings
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('sparse_container', [*CSR_CONTAINERS, *CSC_CONTAINERS])
def test_kernel_ridge_sparse(sparse_container):
    X_sparse = sparse_container(X)
    pred = Ridge(alpha=1, fit_intercept=False, solver='cholesky').fit(X_sparse, y).predict(X_sparse)
    pred2 = KernelRidge(kernel='linear', alpha=1).fit(X_sparse, y).predict(X_sparse)
    assert_array_almost_equal(pred, pred2)