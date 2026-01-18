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
def test_min_max_axis_errors(csc_container, csr_container):
    X = np.array([[0, 3, 0], [2, -1, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64)
    X_csr = csr_container(X)
    X_csc = csc_container(X)
    with pytest.raises(TypeError):
        min_max_axis(X_csr.tolil(), axis=0)
    with pytest.raises(ValueError):
        min_max_axis(X_csr, axis=2)
    with pytest.raises(ValueError):
        min_max_axis(X_csc, axis=-3)