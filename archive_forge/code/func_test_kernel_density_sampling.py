import joblib
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors
from sklearn.neighbors._ball_tree import kernel_norm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_allclose
def test_kernel_density_sampling(n_samples=100, n_features=3):
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    bandwidth = 0.2
    for kernel in ['gaussian', 'tophat']:
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
        samp = kde.sample(100)
        assert X.shape == samp.shape
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)
        dist, ind = nbrs.kneighbors(X, return_distance=True)
        if kernel == 'tophat':
            assert np.all(dist < bandwidth)
        elif kernel == 'gaussian':
            assert np.all(dist < 5 * bandwidth)
    for kernel in ['epanechnikov', 'exponential', 'linear', 'cosine']:
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
        with pytest.raises(NotImplementedError):
            kde.sample(100)
    X = rng.randn(4, 1)
    kde = KernelDensity(kernel='gaussian').fit(X)
    assert kde.sample().shape == (1, 1)