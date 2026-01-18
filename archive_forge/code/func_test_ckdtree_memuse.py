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
@pytest.mark.skipif(python_implementation() == 'PyPy', reason='Fails on PyPy CI runs. See #9507')
def test_ckdtree_memuse():
    try:
        import resource
    except ImportError:
        return
    dx, dy = (0.05, 0.05)
    y, x = np.mgrid[slice(1, 5 + dy, dy), slice(1, 5 + dx, dx)]
    z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
    z_copy = np.empty_like(z)
    z_copy[:] = z
    FILLVAL = 99.0
    mask = np.random.randint(0, z.size, np.random.randint(50) + 5)
    z_copy.flat[mask] = FILLVAL
    igood = np.vstack(np.nonzero(x != FILLVAL)).T
    ibad = np.vstack(np.nonzero(x == FILLVAL)).T
    mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for i in range(10):
        tree = cKDTree(igood)
    num_leaks = 0
    for i in range(100):
        mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tree = cKDTree(igood)
        dist, iquery = tree.query(ibad, k=4, p=2)
        new_mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if new_mem_use > mem_use:
            num_leaks += 1
    assert_(num_leaks < 10)