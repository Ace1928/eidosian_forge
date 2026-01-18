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
def test_distance_vectorization():
    np.random.seed(1234)
    x = np.random.randn(10, 1, 3)
    y = np.random.randn(1, 7, 3)
    assert_equal(minkowski_distance(x, y).shape, (10, 7))