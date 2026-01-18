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
def simulate_periodic_box(kdtree, data, k, boxsize, p):
    dd = []
    ii = []
    x = np.arange(3 ** data.shape[1])
    nn = np.array(np.unravel_index(x, [3] * data.shape[1])).T
    nn = nn - 1.0
    for n in nn:
        image = data + n * 1.0 * boxsize
        dd2, ii2 = kdtree.query(image, k, p=p)
        dd2 = dd2.reshape(-1, k)
        ii2 = ii2.reshape(-1, k)
        dd.append(dd2)
        ii.append(ii2)
    dd = np.concatenate(dd, axis=-1)
    ii = np.concatenate(ii, axis=-1)
    result = np.empty([len(data), len(nn) * k], dtype=[('ii', 'i8'), ('dd', 'f8')])
    result['ii'][:] = ii
    result['dd'][:] = dd
    result.sort(order='dd')
    return (result['dd'][:, :k], result['ii'][:, :k])