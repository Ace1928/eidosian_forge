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
@pytest.mark.parametrize('metric', ['manhattan', 'euclidean'])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_memmap_backed_data(metric, Dispatcher, dtype):
    """Check that the results do not depend on the datasets writability."""
    rng = np.random.RandomState(0)
    spread = 100
    n_samples, n_features = (128, 10)
    X = rng.rand(n_samples, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples, n_features).astype(dtype) * spread
    X_mm, Y_mm = create_memmap_backed_data([X, Y])
    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        radius = 10 ** np.log(n_features)
        parameter = radius
        check_parameters = {'radius': radius}
        compute_parameters = {'sort_results': True}
    ref_dist, ref_indices = Dispatcher.compute(X, Y, parameter, metric=metric, return_distance=True, **compute_parameters)
    dist_mm, indices_mm = Dispatcher.compute(X_mm, Y_mm, parameter, metric=metric, return_distance=True, **compute_parameters)
    ASSERT_RESULT[Dispatcher, dtype](ref_dist, dist_mm, ref_indices, indices_mm, **check_parameters)