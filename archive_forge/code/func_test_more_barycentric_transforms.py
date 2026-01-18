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
@pytest.mark.slow
def test_more_barycentric_transforms(self):
    eps = np.finfo(float).eps
    npoints = {2: 70, 3: 11, 4: 5, 5: 3}
    for ndim in range(2, 6):
        x = np.linspace(0, 1, npoints[ndim])
        grid = np.c_[list(map(np.ravel, np.broadcast_arrays(*np.ix_(*[x] * ndim))))].T
        err_msg = 'ndim=%d' % ndim
        tri = qhull.Delaunay(grid)
        self._check_barycentric_transforms(tri, err_msg=err_msg, unit_cube=True)
        np.random.seed(1234)
        m = np.random.rand(grid.shape[0]) < 0.2
        grid[m, :] += 2 * eps * (np.random.rand(*grid[m, :].shape) - 0.5)
        tri = qhull.Delaunay(grid)
        self._check_barycentric_transforms(tri, err_msg=err_msg, unit_cube=True, unit_cube_tol=2 * eps)
        tri = qhull.Delaunay(np.r_[grid, grid])
        self._check_barycentric_transforms(tri, err_msg=err_msg, unit_cube=True, unit_cube_tol=2 * eps)