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
@pytest.mark.parametrize('Dispatcher, dtype', [(ArgKmin, np.float64), (RadiusNeighbors, np.float32), (ArgKmin, np.float32), (RadiusNeighbors, np.float64)])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_format_agnosticism(global_random_seed, Dispatcher, dtype, csr_container):
    """Check that results do not depend on the format (dense, sparse) of the input."""
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    n_samples, n_features = (100, 100)
    X = rng.rand(n_samples, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples, n_features).astype(dtype) * spread
    X_csr = csr_container(X)
    Y_csr = csr_container(Y)
    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        radius = _non_trivial_radius(X=X, Y=Y, metric='euclidean')
        parameter = radius
        check_parameters = {'radius': radius}
        compute_parameters = {'sort_results': True}
    dist_dense, indices_dense = Dispatcher.compute(X, Y, parameter, chunk_size=50, return_distance=True, **compute_parameters)
    for _X, _Y in itertools.product((X, X_csr), (Y, Y_csr)):
        if _X is X and _Y is Y:
            continue
        dist, indices = Dispatcher.compute(_X, _Y, parameter, chunk_size=50, return_distance=True, **compute_parameters)
        ASSERT_RESULT[Dispatcher, dtype](dist_dense, dist, indices_dense, indices, **check_parameters)