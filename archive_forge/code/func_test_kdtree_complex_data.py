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
def test_kdtree_complex_data():
    points = np.random.rand(10, 2).view(complex)
    with pytest.raises(TypeError, match='complex data'):
        t = KDTree(points)
    t = KDTree(points.real)
    with pytest.raises(TypeError, match='complex data'):
        t.query(points)
    with pytest.raises(TypeError, match='complex data'):
        t.query_ball_point(points, r=1)