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
@requires_cftime
@requires_dask
@pytest.mark.parametrize(('units', 'dtype'), [('days since 1700-01-01', np.dtype('int32')), (None, None)])
def test_encode_cf_datetime_cftime_datetime_via_dask(units, dtype) -> None:
    import dask.array
    calendar = 'standard'
    times = cftime_range(start='1700', freq='D', periods=3, calendar=calendar)
    times = dask.array.from_array(times, chunks=1)
    encoded_times, encoding_units, encoding_calendar = encode_cf_datetime(times, units, None, dtype)
    assert is_duck_dask_array(encoded_times)
    assert encoded_times.chunks == times.chunks
    if units is not None and dtype is not None:
        assert encoding_units == units
        assert encoded_times.dtype == dtype
    else:
        assert encoding_units == 'microseconds since 1970-01-01'
        assert encoded_times.dtype == np.int64
    assert encoding_calendar == calendar
    decoded_times = decode_cf_datetime(encoded_times, encoding_units, encoding_calendar, use_cftime=True)
    np.testing.assert_equal(decoded_times, times)