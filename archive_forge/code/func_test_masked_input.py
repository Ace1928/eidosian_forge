from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
import scipy.sparse
from scipy.io import mmread
import pytest
def test_masked_input():
    np.ma.masked_equal(directed_G, 0)

    def check(method):
        SP = shortest_path(directed_G, method=method, directed=True, overwrite=False)
        assert_array_almost_equal(SP, directed_SP)
    for method in methods:
        check(method)