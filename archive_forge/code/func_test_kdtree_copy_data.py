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
def test_kdtree_copy_data(kdtree_type):
    np.random.seed(0)
    n = 5000
    k = 4
    points = np.random.randn(n, k)
    T = kdtree_type(points, copy_data=True)
    q = points.copy()
    T1 = T.query(q, k=5)[-1]
    points[...] = np.random.randn(n, k)
    T2 = T.query(q, k=5)[-1]
    assert_array_equal(T1, T2)