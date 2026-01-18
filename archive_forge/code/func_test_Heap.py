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
def test_Heap():
    values = np.array([2, -1, 0, -1.5, 3])
    heap = Heap(values)
    pair = heap.get_min()
    assert_equal(pair['key'], 3)
    assert_equal(pair['value'], -1.5)
    heap.remove_min()
    pair = heap.get_min()
    assert_equal(pair['key'], 1)
    assert_equal(pair['value'], -1)
    heap.change_value(1, 2.5)
    pair = heap.get_min()
    assert_equal(pair['key'], 2)
    assert_equal(pair['value'], 0)
    heap.remove_min()
    heap.remove_min()
    heap.change_value(1, 10)
    pair = heap.get_min()
    assert_equal(pair['key'], 4)
    assert_equal(pair['value'], 3)
    heap.remove_min()
    pair = heap.get_min()
    assert_equal(pair['key'], 1)
    assert_equal(pair['value'], 10)