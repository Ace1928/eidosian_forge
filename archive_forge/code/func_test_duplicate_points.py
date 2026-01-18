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
def test_duplicate_points(self):
    x = np.array([0, 1, 0, 1], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    xp = np.r_[x, x]
    yp = np.r_[y, y]
    qhull.Delaunay(np.c_[x, y])
    qhull.Delaunay(np.c_[xp, yp])