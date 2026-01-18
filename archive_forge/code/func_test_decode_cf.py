from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
@pytest.mark.parametrize('calendar', _ALL_CALENDARS)
def test_decode_cf(calendar) -> None:
    days = [1.0, 2.0, 3.0]
    da = DataArray(days, coords=[days], dims=['time'], name='test')
    ds = da.to_dataset()
    for v in ['test', 'time']:
        ds[v].attrs['units'] = 'days since 2001-01-01'
        ds[v].attrs['calendar'] = calendar
    if not has_cftime and calendar not in _STANDARD_CALENDARS:
        with pytest.raises(ValueError):
            ds = decode_cf(ds)
    else:
        ds = decode_cf(ds)
        if calendar not in _STANDARD_CALENDARS:
            assert ds.test.dtype == np.dtype('O')
        else:
            assert ds.test.dtype == np.dtype('M8[ns]')