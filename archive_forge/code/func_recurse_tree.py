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
def recurse_tree(n):
    assert_(isinstance(n, cKDTreeNode))
    if n.split_dim == -1:
        assert_(n.lesser is None)
        assert_(n.greater is None)
        assert_(n.indices.shape[0] <= kdtree.leafsize)
    else:
        recurse_tree(n.lesser)
        recurse_tree(n.greater)
        x = n.lesser.data_points[:, n.split_dim]
        y = n.greater.data_points[:, n.split_dim]
        assert_(x.max() < y.min())