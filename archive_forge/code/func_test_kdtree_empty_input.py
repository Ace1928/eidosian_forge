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
@pytest.mark.parametrize('balanced_tree, compact_nodes', [(True, False), (True, True), (False, False), (False, True)])
def test_kdtree_empty_input(kdtree_type, balanced_tree, compact_nodes):
    np.random.seed(1234)
    empty_v3 = np.empty(shape=(0, 3))
    query_v3 = np.ones(shape=(1, 3))
    query_v2 = np.ones(shape=(2, 3))
    tree = kdtree_type(empty_v3, balanced_tree=balanced_tree, compact_nodes=compact_nodes)
    length = tree.query_ball_point(query_v3, 0.3, return_length=True)
    assert length == 0
    dd, ii = tree.query(query_v2, 2)
    assert ii.shape == (2, 2)
    assert dd.shape == (2, 2)
    assert np.isinf(dd).all()
    N = tree.count_neighbors(tree, [0, 1])
    assert_array_equal(N, [0, 0])
    M = tree.sparse_distance_matrix(tree, 0.3)
    assert M.shape == (0, 0)