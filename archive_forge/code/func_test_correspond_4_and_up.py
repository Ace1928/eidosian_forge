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
def test_correspond_4_and_up(self, xp):
    for i, j in list(zip(list(range(2, 4)), list(range(3, 5)))) + list(zip(list(range(3, 5)), list(range(2, 4)))):
        y = np.random.rand(i * (i - 1) // 2)
        y2 = np.random.rand(j * (j - 1) // 2)
        y = xp.asarray(y)
        y2 = xp.asarray(y2)
        Z = linkage(y)
        Z2 = linkage(y2)
        assert not correspond(Z, y2)
        assert not correspond(Z2, y)