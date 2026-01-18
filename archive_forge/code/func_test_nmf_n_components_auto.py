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
@pytest.mark.parametrize('Estimator', [NMF, MiniBatchNMF])
def test_nmf_n_components_auto(Estimator):
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    W = rng.random_sample((6, 2))
    H = rng.random_sample((2, 5))
    est = Estimator(n_components='auto', init='custom', random_state=0, tol=1e-06)
    est.fit_transform(X, W=W, H=H)
    assert est._n_components == H.shape[0]