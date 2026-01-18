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
def test_kdtree_count_neighbous_multiple_r(kdtree_type):
    n = 2000
    m = 2
    np.random.seed(1234)
    data = np.random.normal(size=(n, m))
    kdtree = kdtree_type(data, leafsize=1)
    r0 = [0, 0.01, 0.01, 0.02, 0.05]
    i0 = np.arange(len(r0))
    n0 = kdtree.count_neighbors(kdtree, r0)
    nnc = kdtree.count_neighbors(kdtree, r0, cumulative=False)
    assert_equal(n0, nnc.cumsum())
    for i, r in zip(itertools.permutations(i0), itertools.permutations(r0)):
        n = kdtree.count_neighbors(kdtree, r)
        assert_array_equal(n, n0[list(i)])