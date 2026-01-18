import itertools
import os
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('whiten', ['arbitrary-variance', 'unit-variance', False])
@pytest.mark.parametrize('return_X_mean', [True, False])
@pytest.mark.parametrize('return_n_iter', [True, False])
def test_fastica_output_shape(whiten, return_X_mean, return_n_iter):
    n_features = 3
    n_samples = 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    expected_len = 3 + return_X_mean + return_n_iter
    out = fastica(X, whiten=whiten, return_n_iter=return_n_iter, return_X_mean=return_X_mean)
    assert len(out) == expected_len
    if not whiten:
        assert out[0] is None