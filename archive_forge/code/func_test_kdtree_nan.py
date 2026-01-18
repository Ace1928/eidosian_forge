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
def test_kdtree_nan():
    vals = [1, 5, -10, 7, -4, -16, -6, 6, 3, -11]
    n = len(vals)
    data = np.concatenate([vals, np.full(n, np.nan)])[:, None]
    with pytest.raises(ValueError, match='must be finite'):
        KDTree(data)