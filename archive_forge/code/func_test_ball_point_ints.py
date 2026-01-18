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
def test_ball_point_ints(kdtree_type):
    x, y = np.mgrid[0:4, 0:4]
    points = list(zip(x.ravel(), y.ravel()))
    tree = kdtree_type(points)
    assert_equal(sorted([4, 8, 9, 12]), sorted(tree.query_ball_point((2, 0), 1)))
    points = np.asarray(points, dtype=float)
    tree = kdtree_type(points)
    assert_equal(sorted([4, 8, 9, 12]), sorted(tree.query_ball_point((2, 0), 1)))