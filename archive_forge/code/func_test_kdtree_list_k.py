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
def test_kdtree_list_k(kdtree_type):
    n = 200
    m = 2
    klist = [1, 2, 3]
    kint = 3
    np.random.seed(1234)
    data = np.random.uniform(size=(n, m))
    kdtree = kdtree_type(data, leafsize=1)
    dd, ii = kdtree.query(data, klist)
    dd1, ii1 = kdtree.query(data, kint)
    assert_equal(dd, dd1)
    assert_equal(ii, ii1)
    klist = np.array([1, 3])
    kint = 3
    dd, ii = kdtree.query(data, kint)
    dd1, ii1 = kdtree.query(data, klist)
    assert_equal(dd1, dd[..., klist - 1])
    assert_equal(ii1, ii[..., klist - 1])
    dd, ii = kdtree.query(data, 1)
    dd1, ii1 = kdtree.query(data, [1])
    assert_equal(len(dd.shape), 1)
    assert_equal(len(dd1.shape), 2)
    assert_equal(dd, np.ravel(dd1))
    assert_equal(ii, np.ravel(ii1))