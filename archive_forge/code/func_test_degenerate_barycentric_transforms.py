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
def test_degenerate_barycentric_transforms(self):
    data = np.load(os.path.join(os.path.dirname(__file__), 'data', 'degenerate_pointset.npz'))
    points = data['c']
    data.close()
    tri = qhull.Delaunay(points)
    bad_count = np.isnan(tri.transform[:, 0, 0]).sum()
    assert_(bad_count < 23, bad_count)
    self._check_barycentric_transforms(tri)