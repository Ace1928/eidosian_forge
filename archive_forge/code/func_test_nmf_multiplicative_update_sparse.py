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
@ignore_warnings(category=ConvergenceWarning)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_nmf_multiplicative_update_sparse(csr_container):
    n_samples = 20
    n_features = 10
    n_components = 5
    alpha = 0.1
    l1_ratio = 0.5
    n_iter = 20
    rng = np.random.mtrand.RandomState(1337)
    X = rng.randn(n_samples, n_features)
    X = np.abs(X)
    X_csr = csr_container(X)
    W0, H0 = nmf._initialize_nmf(X, n_components, init='random', random_state=42)
    for beta_loss in (-1.2, 0, 0.2, 1.0, 2.0, 2.5):
        W, H = (W0.copy(), H0.copy())
        W1, H1, _ = non_negative_factorization(X, W, H, n_components, init='custom', update_H=True, solver='mu', beta_loss=beta_loss, max_iter=n_iter, alpha_W=alpha, l1_ratio=l1_ratio, random_state=42)
        W, H = (W0.copy(), H0.copy())
        W2, H2, _ = non_negative_factorization(X_csr, W, H, n_components, init='custom', update_H=True, solver='mu', beta_loss=beta_loss, max_iter=n_iter, alpha_W=alpha, l1_ratio=l1_ratio, random_state=42)
        assert_allclose(W1, W2, atol=1e-07)
        assert_allclose(H1, H2, atol=1e-07)
        beta_loss -= 1e-05
        W, H = (W0.copy(), H0.copy())
        W3, H3, _ = non_negative_factorization(X_csr, W, H, n_components, init='custom', update_H=True, solver='mu', beta_loss=beta_loss, max_iter=n_iter, alpha_W=alpha, l1_ratio=l1_ratio, random_state=42)
        assert_allclose(W1, W3, atol=0.0001)
        assert_allclose(H1, H3, atol=0.0001)