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
def check_leaves_list_Q(self, method, xp):
    X = xp.asarray(hierarchy_test_data.Q_X)
    Z = linkage(X, method)
    node = to_tree(Z)
    assert_allclose(node.pre_order(), leaves_list(Z), rtol=1e-15)