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
@pytest.mark.parametrize('kdtree_class', [KDTree, cKDTree])
def test_kdtree_count_neighbors_weighted(kdtree_class):
    np.random.seed(1234)
    r = np.arange(0.05, 1, 0.05)
    A = np.random.random(21).reshape((7, 3))
    B = np.random.random(45).reshape((15, 3))
    wA = np.random.random(7)
    wB = np.random.random(15)
    kdA = kdtree_class(A)
    kdB = kdtree_class(B)
    nAB = kdA.count_neighbors(kdB, r, cumulative=False, weights=(wA, wB))
    weights = wA[None, :] * wB[:, None]
    dist = np.linalg.norm(A[None, :, :] - B[:, None, :], axis=-1)
    expect = [np.sum(weights[(prev_radius < dist) & (dist <= radius)]) for prev_radius, radius in zip(itertools.chain([0], r[:-1]), r)]
    assert_allclose(nAB, expect)