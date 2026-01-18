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
@pytest.mark.parametrize('X1, X2', [(sp.random(5, 2, density=0.8, format='csr', random_state=0), sp.random(13, 2, density=0.8, format='csr', random_state=0)), (sp.random(5, 2, density=0.8, format='csr', random_state=0), sp.hstack([np.full((13, 1), fill_value=np.nan), sp.random(13, 1, density=0.8, random_state=42)], format='csr'))])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_incr_mean_variance_axis_equivalence_mean_variance(X1, X2, csr_container):
    X1 = csr_container(X1)
    X2 = csr_container(X2)
    axis = 0
    last_mean, last_var = (np.zeros(X1.shape[1]), np.zeros(X1.shape[1]))
    last_n = np.zeros(X1.shape[1], dtype=np.int64)
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(X1, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n)
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(X2, axis=axis, last_mean=updated_mean, last_var=updated_var, last_n=updated_n)
    X = sp.vstack([X1, X2])
    assert_allclose(updated_mean, np.nanmean(X.toarray(), axis=axis))
    assert_allclose(updated_var, np.nanvar(X.toarray(), axis=axis))
    assert_allclose(updated_n, np.count_nonzero(~np.isnan(X.toarray()), axis=0))