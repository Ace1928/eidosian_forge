import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
def test_laplacian_value_error():
    for t in (int, float, complex):
        for m in ([1, 1], [[[1]]], [[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4], [5, 5]]):
            A = np.array(m, dtype=t)
            assert_raises(ValueError, csgraph.laplacian, A)