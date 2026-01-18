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
def test_coplanar(self):
    points = np.random.rand(10, 2)
    points = np.r_[points, points]
    tri = qhull.Delaunay(points)
    assert_(len(np.unique(tri.simplices.ravel())) == len(points) // 2)
    assert_(len(tri.coplanar) == len(points) // 2)
    assert_(len(np.unique(tri.coplanar[:, 2])) == len(points) // 2)
    assert_(np.all(tri.vertex_to_simplex >= 0))