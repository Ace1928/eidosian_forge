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
def test_nmf_true_reconstruction():
    n_samples = 15
    n_features = 10
    n_components = 5
    beta_loss = 1
    batch_size = 3
    max_iter = 1000
    rng = np.random.mtrand.RandomState(42)
    W_true = np.zeros([n_samples, n_components])
    W_array = np.abs(rng.randn(n_samples))
    for j in range(n_components):
        W_true[j % n_samples, j] = W_array[j % n_samples]
    H_true = np.zeros([n_components, n_features])
    H_array = np.abs(rng.randn(n_components))
    for j in range(n_features):
        H_true[j % n_components, j] = H_array[j % n_components]
    X = np.dot(W_true, H_true)
    model = NMF(n_components=n_components, solver='mu', beta_loss=beta_loss, max_iter=max_iter, random_state=0)
    transf = model.fit_transform(X)
    X_calc = np.dot(transf, model.components_)
    assert model.reconstruction_err_ < 0.1
    assert_allclose(X, X_calc)
    mbmodel = MiniBatchNMF(n_components=n_components, beta_loss=beta_loss, batch_size=batch_size, random_state=0, max_iter=max_iter)
    transf = mbmodel.fit_transform(X)
    X_calc = np.dot(transf, mbmodel.components_)
    assert mbmodel.reconstruction_err_ < 0.1
    assert_allclose(X, X_calc, atol=1)