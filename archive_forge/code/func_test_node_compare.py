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
def test_node_compare(xp):
    np.random.seed(23)
    nobs = 50
    X = np.random.randn(nobs, 4)
    X = xp.asarray(X)
    Z = scipy.cluster.hierarchy.ward(X)
    tree = to_tree(Z)
    assert_(tree > tree.get_left())
    assert_(tree.get_right() > tree.get_left())
    assert_(tree.get_right() == tree.get_right())
    assert_(tree.get_right() != tree.get_left())