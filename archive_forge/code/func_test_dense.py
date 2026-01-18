import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_dense(self):
    funcs = [lambda x, y: x ** 2, lambda x, y: y ** 2, lambda x, y: x ** 2 - y ** 2, lambda x, y: x * y, lambda x, y: np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)]
    np.random.seed(4321)
    grid = np.r_[np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=float), np.random.rand(30 * 30, 2)]
    for j, func in enumerate(funcs):
        self._check_accuracy(func, x=grid, tol=1e-09, atol=0.005, rtol=0.01, err_msg='Function %d' % j)
        self._check_accuracy(func, x=grid, tol=1e-09, atol=0.005, rtol=0.01, err_msg='Function %d' % j, rescale=True)