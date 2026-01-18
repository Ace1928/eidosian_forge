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
@skip_if_array_api
def test_is_isomorphic_4C(self):
    a = np.asarray([7, 2, 3])
    b = np.asarray([6, 3, 2])
    assert_(is_isomorphic(a, b))
    assert_(is_isomorphic(b, a))