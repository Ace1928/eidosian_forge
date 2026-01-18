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
@array_api_compatible
def test_euclidean_linkage_value_error(xp):
    for method in scipy.cluster.hierarchy._EUCLIDEAN_METHODS:
        assert_raises(ValueError, linkage, xp.asarray([[1, 1], [1, 1]]), method=method, metric='cityblock')