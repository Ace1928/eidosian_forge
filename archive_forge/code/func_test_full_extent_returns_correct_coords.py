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
def test_full_extent_returns_correct_coords():
    with open_rasterio(TEST_RASTER_PATH) as src:
        res = ds.utils.calc_res(src)
        left, bottom, right, top = ds.utils.calc_bbox(src.x.values, src.y.values, res)
        cvs = ds.Canvas(plot_width=512, plot_height=256, x_range=[left, right], y_range=[bottom, top])
        agg = cvs.raster(src)
        assert agg.shape == (3, 256, 512)
        assert agg is not None
        for dim in src.dims:
            assert np.all(agg[dim].data == src[dim].data)
        assert np.allclose(agg.x_range, (-180, 180))
        assert np.allclose(agg.y_range, (-90, 90))