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
def test_in_ball_outside(self):
    l = self.T.query_ball_point(self.x + 1.0, self.d, p=self.p, eps=self.eps)
    for i in l:
        assert_(self.distance(self.data[i], self.x, self.p) <= self.d * (1.0 + self.eps))
    l = self.T.query_ball_point(self.x - 1.0, self.d, p=self.p, eps=self.eps)
    for i in l:
        assert_(self.distance(self.data[i], self.x, self.p) <= self.d * (1.0 + self.eps))