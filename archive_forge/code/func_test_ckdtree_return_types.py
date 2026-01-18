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
def test_ckdtree_return_types(self):
    ref = np.zeros((self.n, self.n))
    for i in range(self.n):
        for j in range(self.n):
            v = self.data1[i, :] - self.data2[j, :]
            ref[i, j] = np.dot(v, v)
    ref = np.sqrt(ref)
    ref[ref > self.r] = 0.0
    dist = np.zeros((self.n, self.n))
    r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='dict')
    for i, j in r.keys():
        dist[i, j] = r[i, j]
    assert_array_almost_equal(ref, dist, decimal=14)
    dist = np.zeros((self.n, self.n))
    r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='ndarray')
    for k in range(r.shape[0]):
        i = r['i'][k]
        j = r['j'][k]
        v = r['v'][k]
        dist[i, j] = v
    assert_array_almost_equal(ref, dist, decimal=14)
    r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='dok_matrix')
    assert_array_almost_equal(ref, r.toarray(), decimal=14)
    r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='coo_matrix')
    assert_array_almost_equal(ref, r.toarray(), decimal=14)