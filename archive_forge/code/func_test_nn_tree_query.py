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
@pytest.mark.parametrize('Cls, metric', itertools.chain([(KDTree, metric) for metric in KD_TREE_METRICS], [(BallTree, metric) for metric in BALL_TREE_METRICS]))
@pytest.mark.parametrize('k', (1, 3, 5))
@pytest.mark.parametrize('dualtree', (True, False))
@pytest.mark.parametrize('breadth_first', (True, False))
def test_nn_tree_query(Cls, metric, k, dualtree, breadth_first):
    rng = check_random_state(0)
    X = rng.random_sample((40, DIMENSION))
    Y = rng.random_sample((10, DIMENSION))
    kwargs = METRICS[metric]
    kdt = Cls(X, leaf_size=1, metric=metric, **kwargs)
    dist1, ind1 = kdt.query(Y, k, dualtree=dualtree, breadth_first=breadth_first)
    dist2, ind2 = brute_force_neighbors(X, Y, k, metric, **kwargs)
    assert_array_almost_equal(dist1, dist2)