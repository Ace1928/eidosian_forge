from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.core import duck_array_ops
from xarray.tests import (
def test_coarsen_construct_keeps_all_coords(self):
    da = xr.DataArray(np.arange(24), dims=['time'])
    da = da.assign_coords(day=365 * da)
    result = da.coarsen(time=12).construct(time=('year', 'month'))
    assert list(da.coords) == list(result.coords)
    ds = da.to_dataset(name='T')
    result = ds.coarsen(time=12).construct(time=('year', 'month'))
    assert list(da.coords) == list(result.coords)