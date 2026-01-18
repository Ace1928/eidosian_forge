import warnings
import sys
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
from scipy.cluster import _vq
from scipy.conftest import (
from scipy.sparse._sputils import matrix
from scipy._lib._array_api import (
@skip_if_array_api_gpu
@array_api_compatible
def test_kmeans2_kpp_high_dim(self, xp):
    n_dim = 100
    size = 10
    centers = np.vstack([5 * np.ones(n_dim), -5 * np.ones(n_dim)])
    np.random.seed(42)
    data = np.vstack([np.random.multivariate_normal(centers[0], np.eye(n_dim), size=size), np.random.multivariate_normal(centers[1], np.eye(n_dim), size=size)])
    data = xp.asarray(data)
    res, _ = kmeans2(data, xp.asarray(2), minit='++')
    xp_assert_equal(xp.sign(res), xp.sign(xp.asarray(centers)))