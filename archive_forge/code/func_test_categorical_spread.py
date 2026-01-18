from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_categorical_spread():
    a_data = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='int32')
    b_data = np.array([[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='int32')
    c_data = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 0]], dtype='int32')
    data = np.dstack([a_data, b_data, c_data])
    coords = [np.arange(5), np.arange(5)]
    arr = xr.DataArray(data, coords=coords + [['a', 'b', 'c']], dims=dims + ['cat'])
    s = tf.spread(arr)
    o = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    np.testing.assert_equal(s.sel(cat='a').data, o)
    o = np.array([[2, 2, 2, 0, 0], [2, 2, 2, 0, 0], [2, 2, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    np.testing.assert_equal(s.sel(cat='b').data, o)
    o = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 3, 3, 3], [0, 0, 3, 3, 3], [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.sel(cat='c').data, o)