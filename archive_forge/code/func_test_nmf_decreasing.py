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
@pytest.mark.parametrize('solver', ('cd', 'mu'))
def test_nmf_decreasing(solver):
    n_samples = 20
    n_features = 15
    n_components = 10
    alpha = 0.1
    l1_ratio = 0.5
    tol = 0.0
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.abs(X, X)
    W0, H0 = nmf._initialize_nmf(X, n_components, init='random', random_state=42)
    for beta_loss in (-1.2, 0, 0.2, 1.0, 2.0, 2.5):
        if solver != 'mu' and beta_loss != 2:
            continue
        W, H = (W0.copy(), H0.copy())
        previous_loss = None
        for _ in range(30):
            W, H, _ = non_negative_factorization(X, W, H, beta_loss=beta_loss, init='custom', n_components=n_components, max_iter=1, alpha_W=alpha, solver=solver, tol=tol, l1_ratio=l1_ratio, verbose=0, random_state=0, update_H=True)
            loss = nmf._beta_divergence(X, W, H, beta_loss) + alpha * l1_ratio * n_features * W.sum() + alpha * l1_ratio * n_samples * H.sum() + alpha * (1 - l1_ratio) * n_features * (W ** 2).sum() + alpha * (1 - l1_ratio) * n_samples * (H ** 2).sum()
            if previous_loss is not None:
                assert previous_loss > loss
            previous_loss = loss