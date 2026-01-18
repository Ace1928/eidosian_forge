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
def test_leaders_single(self, xp):
    X = hierarchy_test_data.Q_X
    Y = pdist(X)
    Y = xp.asarray(Y)
    Z = linkage(Y)
    T = fcluster(Z, criterion='maxclust', t=3)
    Lright = (xp.asarray([53, 55, 56]), xp.asarray([2, 3, 1]))
    T = xp.asarray(T, dtype=xp.int32)
    L = leaders(Z, T)
    assert_allclose(np.concatenate(L), np.concatenate(Lright), rtol=1e-15)