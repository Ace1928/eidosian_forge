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
@pytest.mark.parametrize(['Estimator', 'solver'], [[NMF, {'solver': 'cd'}], [NMF, {'solver': 'mu'}], [MiniBatchNMF, {}]])
def test_nmf_regularization(Estimator, solver):
    n_samples = 6
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(n_samples, n_features))
    l1_ratio = 1.0
    regul = Estimator(n_components=n_components, alpha_W=0.5, l1_ratio=l1_ratio, random_state=42, **solver)
    model = Estimator(n_components=n_components, alpha_W=0.0, l1_ratio=l1_ratio, random_state=42, **solver)
    W_regul = regul.fit_transform(X)
    W_model = model.fit_transform(X)
    H_regul = regul.components_
    H_model = model.components_
    eps = np.finfo(np.float64).eps
    W_regul_n_zeros = W_regul[W_regul <= eps].size
    W_model_n_zeros = W_model[W_model <= eps].size
    H_regul_n_zeros = H_regul[H_regul <= eps].size
    H_model_n_zeros = H_model[H_model <= eps].size
    assert W_regul_n_zeros > W_model_n_zeros
    assert H_regul_n_zeros > H_model_n_zeros
    l1_ratio = 0.0
    regul = Estimator(n_components=n_components, alpha_W=0.5, l1_ratio=l1_ratio, random_state=42, **solver)
    model = Estimator(n_components=n_components, alpha_W=0.0, l1_ratio=l1_ratio, random_state=42, **solver)
    W_regul = regul.fit_transform(X)
    W_model = model.fit_transform(X)
    H_regul = regul.components_
    H_model = model.components_
    assert linalg.norm(W_model) ** 2.0 + linalg.norm(H_model) ** 2.0 > linalg.norm(W_regul) ** 2.0 + linalg.norm(H_regul) ** 2.0