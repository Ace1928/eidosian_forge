from __future__ import annotations
import pytest
from dask.context import config
from os import path
from itertools import product
import datashader as ds
import xarray as xr
import numpy as np
import dask.array as da
import pandas as pd
from datashader.resampling import compute_chunksize
import datashader.transfer_functions as tf
from packaging.version import Version
@open_rasterio_available
def test_calc_bbox():
    """Assert that bounding boxes are calculated correctly when using the xarray
    rasterio backend.
    """
    import rasterio
    with open_rasterio(TEST_RASTER_PATH) as src:
        xr_res = ds.utils.calc_res(src)
        xr_bounds = ds.utils.calc_bbox(src.x.values, src.y.values, xr_res)
    with rasterio.open(TEST_RASTER_PATH) as src:
        rio_bounds = src.bounds
    assert np.allclose(xr_bounds, rio_bounds, atol=1.0)