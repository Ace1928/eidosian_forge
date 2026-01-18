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
def assert_hulls_equal(points, facets_1, facets_2):
    facets_1 = set(map(sorted_tuple, facets_1))
    facets_2 = set(map(sorted_tuple, facets_2))
    if facets_1 != facets_2 and points.shape[1] == 2:
        eps = 1000 * np.finfo(float).eps
        for a, b in facets_1:
            for ap, bp in facets_2:
                t = points[bp] - points[ap]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                c1 = np.dot(n, points[b] - points[ap])
                c2 = np.dot(n, points[a] - points[ap])
                if not np.allclose(np.dot(c1, n), 0):
                    continue
                if not np.allclose(np.dot(c2, n), 0):
                    continue
                c1 = np.dot(t, points[a] - points[ap])
                c2 = np.dot(t, points[b] - points[ap])
                c3 = np.dot(t, points[bp] - points[ap])
                if c1 < -eps or c1 > c3 + eps:
                    continue
                if c2 < -eps or c2 > c3 + eps:
                    continue
                break
            else:
                raise AssertionError('comparison fails')
        return
    assert_equal(facets_1, facets_2)