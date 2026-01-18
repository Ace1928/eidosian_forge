import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('dim', [2, 3])
def test_linestrings_buffer(dim):
    coords = np.random.randn(10, 3, dim)
    coords1 = np.asarray(coords, order='C')
    result1 = shapely.linestrings(coords1)
    coords2 = np.asarray(coords1, order='F')
    result2 = shapely.linestrings(coords2)
    assert_geometries_equal(result1, result2)
    coords3 = np.asarray(np.swapaxes(np.swapaxes(coords, 0, 2), 1, 0), order='F')
    coords3 = np.swapaxes(np.swapaxes(coords3, 0, 2), 1, 2)
    result3 = shapely.linestrings(coords3)
    assert_geometries_equal(result1, result3)