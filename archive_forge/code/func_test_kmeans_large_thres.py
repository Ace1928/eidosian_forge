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
def test_kmeans_large_thres(self, xp):
    x = xp.asarray([1, 2, 3, 4, 10], dtype=xp.float64)
    res = kmeans(x, xp.asarray(1), thresh=1e+16)
    xp_assert_close(res[0], xp.asarray([4.0], dtype=xp.float64))
    xp_assert_close(res[1], xp.asarray(2.4, dtype=xp.float64)[()])