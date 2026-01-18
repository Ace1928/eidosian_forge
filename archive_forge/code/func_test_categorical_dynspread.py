from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_categorical_dynspread():
    a_data = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='int32')
    b_data = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='int32')
    c_data = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='int32')
    data = np.dstack([a_data, b_data, c_data])
    coords = [np.arange(5), np.arange(5)]
    arr = xr.DataArray(data, coords=coords + [['a', 'b', 'c']], dims=dims + ['cat'])
    assert tf.dynspread(arr, threshold=0.4).equals(tf.spread(arr, 0))
    assert tf.dynspread(arr, threshold=0.7).equals(tf.spread(arr, 1))
    assert tf.dynspread(arr, threshold=1.0).equals(tf.spread(arr, 3))
    assert tf.dynspread(arr, max_px=0).equals(arr)