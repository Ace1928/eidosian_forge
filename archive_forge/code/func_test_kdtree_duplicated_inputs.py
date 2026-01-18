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
def test_kdtree_duplicated_inputs(kdtree_type):
    n = 1024
    for m in range(1, 8):
        data = np.ones((n, m))
        data[n // 2:] = 2
        for balanced, compact in itertools.product((False, True), repeat=2):
            kdtree = kdtree_type(data, balanced_tree=balanced, compact_nodes=compact, leafsize=1)
            assert kdtree.size == 3
            tree = kdtree.tree if kdtree_type is cKDTree else kdtree.tree._node
            assert_equal(np.sort(tree.lesser.indices), np.arange(0, n // 2))
            assert_equal(np.sort(tree.greater.indices), np.arange(n // 2, n))