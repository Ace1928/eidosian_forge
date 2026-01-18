from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
import scipy.sparse
from scipy.io import mmread
import pytest
@pytest.mark.parametrize('n', (10, 100, 1000))
def test_dijkstra_min_only_random(n):
    np.random.seed(1234)
    data = scipy.sparse.rand(n, n, density=0.5, format='lil', random_state=42, dtype=np.float64)
    data.setdiag(np.zeros(n, dtype=np.bool_))
    v = np.arange(n)
    np.random.shuffle(v)
    indices = v[:int(n * 0.1)]
    ds, pred, sources = dijkstra(data, directed=True, indices=indices, min_only=True, return_predecessors=True)
    for k in range(n):
        p = pred[k]
        s = sources[k]
        while p != -9999:
            assert sources[p] == s
            p = pred[p]