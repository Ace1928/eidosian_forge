from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
import scipy.sparse
from scipy.io import mmread
import pytest
def test_construct_shortest_path():

    def check(method, directed):
        SP1, pred = shortest_path(directed_G, directed=directed, overwrite=False, return_predecessors=True)
        SP2 = construct_dist_matrix(directed_G, pred, directed=directed)
        assert_array_almost_equal(SP1, SP2)
    for method in methods:
        for directed in (True, False):
            check(method, directed)