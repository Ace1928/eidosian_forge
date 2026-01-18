import re
import sys
import warnings
from io import StringIO
import numpy as np
import pytest
from scipy import linalg
from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('solver', ('cd', 'mu'))
def test_nmf_inverse_transform(solver):
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 4))
    m = NMF(solver=solver, n_components=4, init='random', random_state=0, max_iter=1000)
    ft = m.fit_transform(A)
    A_new = m.inverse_transform(ft)
    assert_array_almost_equal(A, A_new, decimal=2)