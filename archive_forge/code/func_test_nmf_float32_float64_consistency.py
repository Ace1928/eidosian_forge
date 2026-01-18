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
@pytest.mark.filterwarnings('ignore:The default value of `n_components` will change')
@pytest.mark.parametrize(['Estimator', 'solver'], [[NMF, {'solver': 'cd'}], [NMF, {'solver': 'mu'}], [MiniBatchNMF, {}]])
def test_nmf_float32_float64_consistency(Estimator, solver):
    X = np.random.RandomState(0).randn(50, 7)
    np.abs(X, out=X)
    nmf32 = Estimator(random_state=0, tol=0.001, **solver)
    W32 = nmf32.fit_transform(X.astype(np.float32))
    nmf64 = Estimator(random_state=0, tol=0.001, **solver)
    W64 = nmf64.fit_transform(X)
    assert_allclose(W32, W64, atol=1e-05)