import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
def assert_unordered_allclose(self, arr1, arr2, rtol=1e-07):
    """Check that every line in arr1 is only once in arr2"""
    assert_equal(arr1.shape, arr2.shape)
    truths = np.zeros((arr1.shape[0],), dtype=bool)
    for l1 in arr1:
        indexes = np.nonzero((abs(arr2 - l1) < rtol).all(axis=1))[0]
        assert_equal(indexes.shape, (1,))
        truths[indexes[0]] = True
    assert_(truths.all())