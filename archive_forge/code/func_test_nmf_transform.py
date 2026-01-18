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
@pytest.mark.parametrize('solver', ['cd', 'mu'])
def test_nmf_transform(solver):
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(6, 5))
    m = NMF(solver=solver, n_components=3, init='random', random_state=0, tol=1e-06)
    ft = m.fit_transform(A)
    t = m.transform(A)
    assert_allclose(ft, t, atol=0.1)