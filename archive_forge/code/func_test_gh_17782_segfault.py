from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
import scipy.sparse
from scipy.io import mmread
import pytest
def test_gh_17782_segfault():
    text = '%%MatrixMarket matrix coordinate real general\n                84 84 22\n                2 1 4.699999809265137e+00\n                6 14 1.199999973177910e-01\n                9 6 1.199999973177910e-01\n                10 16 2.012000083923340e+01\n                11 10 1.422000026702881e+01\n                12 1 9.645999908447266e+01\n                13 18 2.012000083923340e+01\n                14 13 4.679999828338623e+00\n                15 11 1.199999973177910e-01\n                16 12 1.199999973177910e-01\n                18 15 1.199999973177910e-01\n                32 2 2.299999952316284e+00\n                33 20 6.000000000000000e+00\n                33 32 5.000000000000000e+00\n                36 9 3.720000028610229e+00\n                36 37 3.720000028610229e+00\n                36 38 3.720000028610229e+00\n                37 44 8.159999847412109e+00\n                38 32 7.903999328613281e+01\n                43 20 2.400000000000000e+01\n                43 33 4.000000000000000e+00\n                44 43 6.028000259399414e+01\n    '
    data = mmread(StringIO(text))
    dijkstra(data, directed=True, return_predecessors=True)