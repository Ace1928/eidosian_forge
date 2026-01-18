from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def assert_eq_xr(agg, b, close=False):
    """Assert that two xarray DataArrays are equal, handling the possibility
    that the two DataArrays may be backed by ndarrays of different types"""
    if cupy:
        if isinstance(agg.data, cupy.ndarray):
            agg = xr.DataArray(cupy.asnumpy(agg.data), coords=agg.coords, dims=agg.dims)
        if isinstance(b.data, cupy.ndarray):
            b = xr.DataArray(cupy.asnumpy(b.data), coords=b.coords, dims=b.dims)
    if close:
        xr.testing.assert_allclose(agg, b)
    else:
        xr.testing.assert_equal(agg, b)