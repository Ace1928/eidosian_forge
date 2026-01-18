from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_cftime
@requires_dask
@pytest.mark.parametrize('use_dask', [False, True])
def test_cftime_round_accessor(cftime_rounding_dataarray, cftime_date_type, use_dask) -> None:
    import dask.array as da
    freq = 'D'
    expected = xr.DataArray([[cftime_date_type(1, 1, 1, 0), cftime_date_type(1, 1, 2, 0)], [cftime_date_type(1, 1, 2, 0), cftime_date_type(1, 1, 2, 0)]], name='round')
    if use_dask:
        chunks = {'dim_0': 1}
        with raise_if_dask_computes(max_computes=1):
            result = cftime_rounding_dataarray.chunk(chunks).dt.round(freq)
        expected = expected.chunk(chunks)
        assert isinstance(result.data, da.Array)
        assert result.chunks == expected.chunks
    else:
        result = cftime_rounding_dataarray.dt.round(freq)
    assert_identical(result, expected)