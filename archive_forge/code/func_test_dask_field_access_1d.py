from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_cftime
@requires_dask
@pytest.mark.parametrize('field', ['year', 'month', 'day', 'hour', 'dayofyear', 'dayofweek'])
def test_dask_field_access_1d(data, field) -> None:
    import dask.array as da
    expected = xr.DataArray(getattr(xr.coding.cftimeindex.CFTimeIndex(data.time.values), field), name=field, dims=['time'])
    times = xr.DataArray(data.time.values, dims=['time']).chunk({'time': 50})
    result = getattr(times.dt, field)
    assert isinstance(result.data, da.Array)
    assert result.chunks == times.chunks
    assert_equal(result.compute(), expected)