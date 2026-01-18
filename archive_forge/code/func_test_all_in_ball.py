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
def test_all_in_ball(self):
    r = self.T1.query_ball_tree(self.T2, self.d, p=self.p, eps=self.eps)
    for i, l in enumerate(r):
        for j in l:
            assert self.distance(self.data1[i], self.data2[j], self.p) <= self.d * (1.0 + self.eps)