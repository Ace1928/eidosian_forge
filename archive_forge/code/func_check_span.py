from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def check_span(x, cmap, how, sol):
    sol = sol.copy()
    if isinstance(x, xr.DataArray) and isinstance(x.data, da.Array):
        x = x.compute()
    else:
        x = x.copy()
    img = tf.shade(x, cmap=cmap, how=how, span=None)
    assert_eq_xr(img, sol)
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    assert_eq_xr(img, sol)
    x[0, 1] = 10
    x_input = x.copy()
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    assert_eq_xr(img, sol)
    x.equals(x_input)
    x[2, 1] = 18
    x_input = x.copy()
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    assert_eq_xr(img, sol)
    x.equals(x_input)
    x[0, 1] = 0 if x.dtype.kind in ('i', 'u') else np.nan
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    sol[0, 1] = sol[0, 0]
    assert_eq_xr(img, sol)
    x[2, 1] = 0 if x.dtype.kind in ('i', 'u') else np.nan
    img = tf.shade(x, cmap=cmap, how=how, span=float_span)
    sol[2, 1] = sol[0, 0]
    assert_eq_xr(img, sol)