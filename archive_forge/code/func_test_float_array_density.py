from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_float_array_density():
    data = np.ones((4, 4), dtype='float32')
    assert tf._array_density(data, float_type=True) == 1.0
    data = np.full((4, 4), np.nan, dtype='float32')
    assert tf._array_density(data, float_type=True) == np.inf
    data[3, 3] = 1
    assert tf._array_density(data, float_type=True) == 0
    data[2, 0] = data[0, 2] = data[1, 1] = 1
    assert np.allclose(tf._array_density(data, float_type=True), 0.75)
    assert np.allclose(tf._array_density(data, float_type=True, px=3), 1)