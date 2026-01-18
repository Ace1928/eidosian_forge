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
def test_convex_hull(self):
    points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=np.float64)
    tri = qhull.Delaunay(points)
    assert_equal(tri.convex_hull, [[3, 2], [1, 2], [1, 0], [3, 0]])