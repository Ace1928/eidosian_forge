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
@pytest.mark.parametrize('inplace_csr_row_normalize', (inplace_csr_row_normalize_l1, inplace_csr_row_normalize_l2))
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_inplace_normalize(csr_container, inplace_csr_row_normalize):
    if csr_container is sp.csr_matrix:
        ones = np.ones((10, 1))
    else:
        ones = np.ones(10)
    rs = RandomState(10)
    for dtype in (np.float64, np.float32):
        X = rs.randn(10, 5).astype(dtype)
        X_csr = csr_container(X)
        for index_dtype in [np.int32, np.int64]:
            if index_dtype is np.int64:
                X_csr.indptr = X_csr.indptr.astype(index_dtype)
                X_csr.indices = X_csr.indices.astype(index_dtype)
            assert X_csr.indices.dtype == index_dtype
            assert X_csr.indptr.dtype == index_dtype
            inplace_csr_row_normalize(X_csr)
            assert X_csr.dtype == dtype
            if inplace_csr_row_normalize is inplace_csr_row_normalize_l2:
                X_csr.data **= 2
            assert_array_almost_equal(np.abs(X_csr).sum(axis=1), ones)