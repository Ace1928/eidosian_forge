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
def test_mean_variance_illegal_axis(csr_container):
    X, _ = make_classification(5, 4, random_state=0)
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_csr = csr_container(X)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=-3)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=2)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=-1)
    with pytest.raises(ValueError):
        incr_mean_variance_axis(X_csr, axis=-3, last_mean=None, last_var=None, last_n=None)
    with pytest.raises(ValueError):
        incr_mean_variance_axis(X_csr, axis=2, last_mean=None, last_var=None, last_n=None)
    with pytest.raises(ValueError):
        incr_mean_variance_axis(X_csr, axis=-1, last_mean=None, last_var=None, last_n=None)