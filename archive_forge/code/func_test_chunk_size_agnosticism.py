import itertools
import re
import warnings
from functools import partial
import numpy as np
import pytest
import threadpoolctl
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('Dispatcher', [ArgKmin, RadiusNeighbors])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_chunk_size_agnosticism(global_random_seed, Dispatcher, dtype, n_features=100):
    """Check that results do not depend on the chunk size."""
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    n_samples_X, n_samples_Y = rng.choice([97, 100, 101, 500], size=2, replace=False)
    X = rng.rand(n_samples_X, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples_Y, n_features).astype(dtype) * spread
    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        radius = _non_trivial_radius(X=X, Y=Y, metric='euclidean')
        parameter = radius
        check_parameters = {'radius': radius}
        compute_parameters = {'sort_results': True}
    ref_dist, ref_indices = Dispatcher.compute(X, Y, parameter, chunk_size=256, metric='manhattan', return_distance=True, **compute_parameters)
    dist, indices = Dispatcher.compute(X, Y, parameter, chunk_size=41, metric='manhattan', return_distance=True, **compute_parameters)
    ASSERT_RESULT[Dispatcher, dtype](ref_dist, dist, ref_indices, indices, **check_parameters)