import itertools
import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.metrics import DistanceMetric
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.utils import check_random_state
@pytest.mark.parametrize('Cls', [KDTree, BallTree])
def test_neighbor_tree_query_radius_distance(Cls, n_samples=100, n_features=10):
    rng = check_random_state(0)
    X = 2 * rng.random_sample(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)
    eps = 1e-15
    tree = Cls(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))
    for r in np.linspace(rad[0], rad[-1], 100):
        ind, dist = tree.query_radius([query_pt], r + eps, return_distance=True)
        ind = ind[0]
        dist = dist[0]
        d = np.sqrt(((query_pt - X[ind]) ** 2).sum(1))
        assert_array_almost_equal(d, dist)