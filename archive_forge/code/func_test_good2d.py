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
@pytest.mark.parametrize('incremental', [False, True])
def test_good2d(self, incremental):
    points = np.array([[0.2, 0.2], [0.2, 0.4], [0.4, 0.4], [0.4, 0.2], [0.3, 0.6]])
    hull = qhull.ConvexHull(points=points, incremental=incremental, qhull_options='QG4')
    expected = np.array([False, True, False, False], dtype=bool)
    actual = hull.good
    assert_equal(actual, expected)