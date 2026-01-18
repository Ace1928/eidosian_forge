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
def test_kmeans_diff_convergence(self, xp):
    obs = xp.asarray([-3, -1, 0, 1, 1, 8], dtype=xp.float64)
    res = kmeans(obs, xp.asarray([-3.0, 0.99]))
    xp_assert_close(res[0], xp.asarray([-0.4, 8.0], dtype=xp.float64))
    xp_assert_close(res[1], xp.asarray(1.0666666666666667, dtype=xp.float64)[()])