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
def test_fastica_simple_different_solvers(add_noise, global_random_seed):
    """Test FastICA is consistent between whiten_solvers."""
    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples, random_state=rng)
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s
    phi = rng.rand() * 2 * np.pi
    mixing = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
    m = np.dot(mixing, s)
    if add_noise:
        m += 0.1 * rng.randn(2, 1000)
    center_and_norm(m)
    outs = {}
    for solver in ('svd', 'eigh'):
        ica = FastICA(random_state=0, whiten='unit-variance', whiten_solver=solver)
        sources = ica.fit_transform(m.T)
        outs[solver] = sources
        assert ica.components_.shape == (2, 2)
        assert sources.shape == (1000, 2)
    assert_allclose(outs['eigh'], outs['svd'], atol=1e-12)