from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
import scipy.sparse
from scipy.io import mmread
import pytest
def test_dijkstra_random():
    n = 10
    indices = [0, 4, 4, 5, 7, 9, 0, 6, 2, 3, 7, 9, 1, 2, 9, 2, 5, 6]
    indptr = [0, 0, 2, 5, 6, 7, 8, 12, 15, 18, 18]
    data = [0.33629, 0.40458, 0.47493, 0.42757, 0.11497, 0.91653, 0.69084, 0.64979, 0.62555, 0.743, 0.01724, 0.99945, 0.31095, 0.15557, 0.02439, 0.65814, 0.23478, 0.24072]
    graph = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n))
    dijkstra(graph, directed=True, return_predecessors=True)