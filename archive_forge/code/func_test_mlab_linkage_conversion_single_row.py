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
def test_mlab_linkage_conversion_single_row(self, xp):
    Z = xp.asarray([[0.0, 1.0, 3.0, 2.0]])
    Zm = xp.asarray([[1, 2, 3]])
    xp_assert_close(from_mlab_linkage(Zm), xp.asarray(Z, dtype=xp.float64), rtol=1e-15)
    xp_assert_close(to_mlab_linkage(Z), xp.asarray(Zm, dtype=xp.float64), rtol=1e-15)