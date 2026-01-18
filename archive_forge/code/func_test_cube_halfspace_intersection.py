import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
@pytest.mark.parametrize('dt', [np.float64, int])
def test_cube_halfspace_intersection(self, dt):
    halfspaces = np.array([[-1, 0, 0], [0, -1, 0], [1, 0, -2], [0, 1, -2]], dtype=dt)
    feasible_point = np.array([1, 1], dtype=dt)
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    hull = qhull.HalfspaceIntersection(halfspaces, feasible_point)
    assert_allclose(hull.intersections, points)