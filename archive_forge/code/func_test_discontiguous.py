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
def test_discontiguous(kdtree_type):
    np.random.seed(1234)
    data = np.random.normal(size=(100, 3))
    d_contiguous = np.arange(100) * 0.04
    d_discontiguous = np.ascontiguousarray(np.arange(100)[::-1] * 0.04)[::-1]
    query_contiguous = np.random.normal(size=(100, 3))
    query_discontiguous = np.ascontiguousarray(query_contiguous.T).T
    assert query_discontiguous.strides[-1] != query_contiguous.strides[-1]
    assert d_discontiguous.strides[-1] != d_contiguous.strides[-1]
    tree = kdtree_type(data)
    length1 = tree.query_ball_point(query_contiguous, d_contiguous, return_length=True)
    length2 = tree.query_ball_point(query_discontiguous, d_discontiguous, return_length=True)
    assert_array_equal(length1, length2)
    d1, i1 = tree.query(query_contiguous, 1)
    d2, i2 = tree.query(query_discontiguous, 1)
    assert_array_equal(d1, d2)
    assert_array_equal(i1, i2)