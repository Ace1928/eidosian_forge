import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_boundary_tri_symmetry(self):
    points = np.array([(0, 0), (1, 0), (0.5, np.sqrt(3) / 2)])
    values = np.array([1, 0, 0])
    ip = interpnd.CloughTocher2DInterpolator(points, values)
    ip.grad[...] = 0
    alpha = 0.3
    p1 = np.array([0.5 * np.cos(alpha), 0.5 * np.sin(alpha)])
    p2 = np.array([0.5 * np.cos(np.pi / 3 - alpha), 0.5 * np.sin(np.pi / 3 - alpha)])
    v1 = ip(p1)
    v2 = ip(p2)
    assert_allclose(v1, v2)
    np.random.seed(1)
    A = np.random.randn(2, 2)
    b = np.random.randn(2)
    points = A.dot(points.T).T + b[None, :]
    p1 = A.dot(p1) + b
    p2 = A.dot(p2) + b
    ip = interpnd.CloughTocher2DInterpolator(points, values)
    ip.grad[...] = 0
    w1 = ip(p1)
    w2 = ip(p2)
    assert_allclose(w1, v1)
    assert_allclose(w2, v2)