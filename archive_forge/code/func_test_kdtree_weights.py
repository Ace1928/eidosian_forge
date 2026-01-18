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
def test_kdtree_weights(kdtree_type):
    data = np.linspace(0, 1, 4).reshape(-1, 1)
    tree1 = kdtree_type(data, leafsize=1)
    weights = np.ones(len(data), dtype='f4')
    nw = tree1._build_weights(weights)
    assert_array_equal(nw, [4, 2, 1, 1, 2, 1, 1])
    assert_raises(ValueError, tree1._build_weights, weights[:-1])
    for i in range(10):
        c1 = tree1.count_neighbors(tree1, np.linspace(0, 10, i))
        c2 = tree1.count_neighbors(tree1, np.linspace(0, 10, i), weights=(weights, weights))
        c3 = tree1.count_neighbors(tree1, np.linspace(0, 10, i), weights=(weights, None))
        c4 = tree1.count_neighbors(tree1, np.linspace(0, 10, i), weights=(None, weights))
        tree1.count_neighbors(tree1, np.linspace(0, 10, i), weights=weights)
        assert_array_equal(c1, c2)
        assert_array_equal(c1, c3)
        assert_array_equal(c1, c4)
    for i in range(len(data)):
        w1 = weights.copy()
        w1[i] = 0
        data2 = data[w1 != 0]
        tree2 = kdtree_type(data2)
        c1 = tree1.count_neighbors(tree1, np.linspace(0, 10, 100), weights=(w1, w1))
        c2 = tree2.count_neighbors(tree2, np.linspace(0, 10, 100))
        assert_array_equal(c1, c2)
        assert_raises(ValueError, tree1.count_neighbors, tree2, np.linspace(0, 10, 100), weights=w1)