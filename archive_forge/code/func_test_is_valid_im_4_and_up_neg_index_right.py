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
def test_is_valid_im_4_and_up_neg_index_right(self, xp):
    for i in range(4, 15, 3):
        y = np.random.rand(i * (i - 1) // 2)
        y = xp.asarray(y)
        Z = linkage(y)
        R = inconsistent(Z)
        R[i // 2, 1] = -2.0
        assert_(is_valid_im(R) is False)
        assert_raises(ValueError, is_valid_im, R, throw=True)