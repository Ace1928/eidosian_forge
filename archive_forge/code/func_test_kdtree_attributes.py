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
def test_kdtree_attributes():
    np.random.seed(1234)
    points = np.random.rand(100, 4)
    t = KDTree(points)
    assert isinstance(t.m, int)
    assert t.n == points.shape[0]
    assert isinstance(t.n, int)
    assert t.m == points.shape[1]
    assert isinstance(t.leafsize, int)
    assert t.leafsize == 10
    assert_array_equal(t.maxes, np.amax(points, axis=0))
    assert_array_equal(t.mins, np.amin(points, axis=0))
    assert t.data is points