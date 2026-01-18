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
@pytest.mark.parametrize('kernel', ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'])
@pytest.mark.parametrize('bandwidth', [0.01, 0.1, 1, 'scott', 'silverman'])
def test_kernel_density(kernel, bandwidth):
    n_samples, n_features = (100, 3)
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    Y = rng.randn(n_samples, n_features)
    dens_true = compute_kernel_slow(Y, X, kernel, bandwidth)
    for rtol in [0, 1e-05]:
        for atol in [1e-06, 0.01]:
            for breadth_first in (True, False):
                check_results(kernel, bandwidth, atol, rtol, X, Y, dens_true)