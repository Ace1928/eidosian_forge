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
@pytest.mark.parametrize('add_noise', [True, False])
def test_non_square_fastica(add_noise):
    rng = np.random.RandomState(0)
    n_samples = 1000
    t = np.linspace(0, 100, n_samples)
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s
    mixing = rng.randn(6, 2)
    m = np.dot(mixing, s)
    if add_noise:
        m += 0.1 * rng.randn(6, n_samples)
    center_and_norm(m)
    k_, mixing_, s_ = fastica(m.T, n_components=2, whiten='unit-variance', random_state=rng)
    s_ = s_.T
    assert_allclose(s_, np.dot(np.dot(mixing_, k_), m))
    center_and_norm(s_)
    s1_, s2_ = s_
    if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
        s2_, s1_ = s_
    s1_ *= np.sign(np.dot(s1_, s1))
    s2_ *= np.sign(np.dot(s2_, s2))
    if not add_noise:
        assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=0.001)
        assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=0.001)