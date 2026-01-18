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
def test_nd_simplex(self):
    for nd in range(2, 8):
        points = np.zeros((nd + 1, nd))
        for j in range(nd):
            points[j, j] = 1.0
        points[-1, :] = 1.0
        tri = qhull.Delaunay(points)
        tri.simplices.sort()
        assert_equal(tri.simplices, np.arange(nd + 1, dtype=int)[None, :])
        assert_equal(tri.neighbors, -1 + np.zeros(nd + 1, dtype=int)[None, :])