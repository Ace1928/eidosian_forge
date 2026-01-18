from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
import scipy.sparse
from scipy.io import mmread
import pytest
def test_NaN_warnings():
    with warnings.catch_warnings(record=True) as record:
        shortest_path(np.array([[0, 1], [np.nan, 0]]))
    for r in record:
        assert r.category is not RuntimeWarning