from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
def test_calendar_datetime64_2d() -> None:
    data = xr.DataArray(np.zeros((4, 5), dtype='datetime64[ns]'), dims=('x', 'y'))
    assert data.dt.calendar == 'proleptic_gregorian'