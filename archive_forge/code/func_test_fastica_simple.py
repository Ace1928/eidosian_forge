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
def test_fastica_simple(add_noise, global_random_seed, global_dtype):
    if global_random_seed == 20 and global_dtype == np.float32 and (not add_noise) and (os.getenv('DISTRIB') == 'ubuntu'):
        pytest.xfail('FastICA instability with Ubuntu Atlas build with float32 global_dtype. For more details, see https://github.com/scikit-learn/scikit-learn/issues/24131#issuecomment-1208091119')
    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples, random_state=global_random_seed)
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s = s.astype(global_dtype)
    s1, s2 = s
    phi = 0.6
    mixing = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
    mixing = mixing.astype(global_dtype)
    m = np.dot(mixing, s)
    if add_noise:
        m += 0.1 * rng.randn(2, 1000)
    center_and_norm(m)

    def g_test(x):
        return (x ** 3, (3 * x ** 2).mean(axis=-1))
    algos = ['parallel', 'deflation']
    nls = ['logcosh', 'exp', 'cube', g_test]
    whitening = ['arbitrary-variance', 'unit-variance', False]
    for algo, nl, whiten in itertools.product(algos, nls, whitening):
        if whiten:
            k_, mixing_, s_ = fastica(m.T, fun=nl, whiten=whiten, algorithm=algo, random_state=rng)
            with pytest.raises(ValueError):
                fastica(m.T, fun=np.tanh, whiten=whiten, algorithm=algo)
        else:
            pca = PCA(n_components=2, whiten=True, random_state=rng)
            X = pca.fit_transform(m.T)
            k_, mixing_, s_ = fastica(X, fun=nl, algorithm=algo, whiten=False, random_state=rng)
            with pytest.raises(ValueError):
                fastica(X, fun=np.tanh, algorithm=algo)
        s_ = s_.T
        if whiten:
            atol = 1e-05 if global_dtype == np.float32 else 0
            assert_allclose(np.dot(np.dot(mixing_, k_), m), s_, atol=atol)
        center_and_norm(s_)
        s1_, s2_ = s_
        if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
            s2_, s1_ = s_
        s1_ *= np.sign(np.dot(s1_, s1))
        s2_ *= np.sign(np.dot(s2_, s2))
        if not add_noise:
            assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=0.01)
            assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=0.01)
        else:
            assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=0.1)
            assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=0.1)
    _, _, sources_fun = fastica(m.T, fun=nl, algorithm=algo, random_state=global_random_seed)
    ica = FastICA(fun=nl, algorithm=algo, random_state=global_random_seed)
    sources = ica.fit_transform(m.T)
    assert ica.components_.shape == (2, 2)
    assert sources.shape == (1000, 2)
    assert_allclose(sources_fun, sources)
    atol = np.max(np.abs(sources)) * (1e-05 if global_dtype == np.float32 else 1e-07)
    assert_allclose(sources, ica.transform(m.T), atol=atol)
    assert ica.mixing_.shape == (2, 2)
    ica = FastICA(fun=np.tanh, algorithm=algo)
    with pytest.raises(ValueError):
        ica.fit(m.T)