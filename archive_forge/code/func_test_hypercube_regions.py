import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
@pytest.mark.parametrize('dim', range(2, 6))
def test_hypercube_regions(self, dim):
    points = np.vstack(list(itertools.product([-1, 1], repeat=dim)))
    points = points.astype(np.float64) / np.sqrt(dim)
    sv = SphericalVoronoi(points)
    expected = np.concatenate((-np.eye(dim), np.eye(dim)))
    dist = distance.cdist(sv.vertices, expected)
    res = linear_sum_assignment(dist)
    assert dist[res].sum() < TOL