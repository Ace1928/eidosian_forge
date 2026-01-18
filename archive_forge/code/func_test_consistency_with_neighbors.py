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
def test_consistency_with_neighbors(self):
    M = self.T1.sparse_distance_matrix(self.T2, self.r)
    r = self.T1.query_ball_tree(self.T2, self.r)
    for i, l in enumerate(r):
        for j in l:
            assert_almost_equal(M[i, j], self.distance(self.T1.data[i], self.T2.data[j], self.p), decimal=14)
    for (i, j), d in M.items():
        assert_(j in r[i])