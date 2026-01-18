from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_dask
@requires_cftime
def test_calendar_dask_cftime() -> None:
    from cftime import num2date
    data = xr.DataArray(num2date(np.random.randint(1, 1000000, size=(4, 5, 6)), 'hours since 1970-01-01T00:00', calendar='noleap'), dims=('x', 'y', 'z')).chunk()
    with raise_if_dask_computes(max_computes=2):
        assert data.dt.calendar == 'noleap'