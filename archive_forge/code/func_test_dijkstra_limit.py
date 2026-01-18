from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
import scipy.sparse
from scipy.io import mmread
import pytest
def test_dijkstra_limit():
    limits = [0, 2, np.inf]
    results = [undirected_SP_limit_0, undirected_SP_limit_2, undirected_SP]

    def check(limit, result):
        SP = dijkstra(undirected_G, directed=False, limit=limit)
        assert_array_almost_equal(SP, result)
    for limit, result in zip(limits, results):
        check(limit, result)