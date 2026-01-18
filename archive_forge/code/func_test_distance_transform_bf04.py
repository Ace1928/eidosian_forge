import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
@pytest.mark.parametrize('dtype', types)
def test_distance_transform_bf04(self, dtype):
    data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
    tdt, tft = ndimage.distance_transform_bf(data, return_indices=1)
    dts = []
    fts = []
    dt = numpy.zeros(data.shape, dtype=numpy.float64)
    ndimage.distance_transform_bf(data, distances=dt)
    dts.append(dt)
    ft = ndimage.distance_transform_bf(data, return_distances=False, return_indices=1)
    fts.append(ft)
    ft = numpy.indices(data.shape, dtype=numpy.int32)
    ndimage.distance_transform_bf(data, return_distances=False, return_indices=True, indices=ft)
    fts.append(ft)
    dt, ft = ndimage.distance_transform_bf(data, return_indices=1)
    dts.append(dt)
    fts.append(ft)
    dt = numpy.zeros(data.shape, dtype=numpy.float64)
    ft = ndimage.distance_transform_bf(data, distances=dt, return_indices=True)
    dts.append(dt)
    fts.append(ft)
    ft = numpy.indices(data.shape, dtype=numpy.int32)
    dt = ndimage.distance_transform_bf(data, return_indices=True, indices=ft)
    dts.append(dt)
    fts.append(ft)
    dt = numpy.zeros(data.shape, dtype=numpy.float64)
    ft = numpy.indices(data.shape, dtype=numpy.int32)
    ndimage.distance_transform_bf(data, distances=dt, return_indices=True, indices=ft)
    dts.append(dt)
    fts.append(ft)
    for dt in dts:
        assert_array_almost_equal(tdt, dt)
    for ft in fts:
        assert_array_almost_equal(tft, ft)