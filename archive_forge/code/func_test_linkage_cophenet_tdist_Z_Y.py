import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close
from . import hierarchy_test_data
@skip_if_array_api_gpu
@array_api_compatible
def test_linkage_cophenet_tdist_Z_Y(self, xp):
    Z = xp.asarray(hierarchy_test_data.linkage_ytdist_single)
    c, M = cophenet(Z, xp.asarray(hierarchy_test_data.ytdist))
    expectedM = xp.asarray([268, 295, 255, 255, 295, 295, 268, 268, 295, 295, 295, 138, 219, 295, 295], dtype=xp.float64)
    expectedc = xp.asarray(0.6399312964333934, dtype=xp.float64)[()]
    xp_assert_close(c, expectedc, atol=1e-10)
    xp_assert_close(M, expectedM, atol=1e-10)