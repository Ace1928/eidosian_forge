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
def test_multiple_radius(self):
    rs = np.exp(np.linspace(np.log(0.01), np.log(10), 3))
    results = self.T1.count_neighbors(self.T2, rs)
    assert_(np.all(np.diff(results) >= 0))
    for r, result in zip(rs, results):
        assert_equal(self.T1.count_neighbors(self.T2, r), result)