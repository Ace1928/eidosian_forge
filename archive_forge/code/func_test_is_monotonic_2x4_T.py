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
def test_is_monotonic_2x4_T(self, xp):
    Z = xp.asarray([[0, 1, 0.3, 2], [2, 3, 0.4, 3]], dtype=xp.float64)
    assert is_monotonic(Z)