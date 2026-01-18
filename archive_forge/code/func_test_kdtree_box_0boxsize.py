import os
from numpy.testing import (assert_equal, assert_array_equal, assert_,
from pytest import raises as assert_raises
import pytest
from platform import python_implementation
import numpy as np
from scipy.spatial import KDTree, Rectangle, distance_matrix, cKDTree
from scipy.spatial._ckdtree import cKDTreeNode
from scipy.spatial import minkowski_distance
import itertools
def test_kdtree_box_0boxsize(kdtree_type):
    n = 2000
    m = 2
    k = 3
    np.random.seed(1234)
    data = np.random.uniform(size=(n, m))
    kdtree = kdtree_type(data, leafsize=1, boxsize=0.0)
    kdtree2 = kdtree_type(data, leafsize=1)
    for p in [1, 2, np.inf]:
        dd, ii = kdtree.query(data, k, p=p)
        dd1, ii1 = kdtree2.query(data, k, p=p)
        assert_almost_equal(dd, dd1)
        assert_equal(ii, ii1)