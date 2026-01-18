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
def test_linkage_cophenet_tdist_Z(self, xp):
    expectedM = xp.asarray([268, 295, 255, 255, 295, 295, 268, 268, 295, 295, 295, 138, 219, 295, 295])
    Z = xp.asarray(hierarchy_test_data.linkage_ytdist_single)
    M = cophenet(Z)
    xp_assert_close(M, xp.asarray(expectedM, dtype=xp.float64), atol=1e-10)