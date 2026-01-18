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
@requires_dask
@pytest.mark.parametrize(('units', 'dtype'), [('days', np.dtype('int32')), (None, None)])
def test_encode_cf_timedelta_via_dask(units, dtype) -> None:
    import dask.array
    times = pd.timedelta_range(start='0D', freq='D', periods=3)
    times = dask.array.from_array(times, chunks=1)
    encoded_times, encoding_units = encode_cf_timedelta(times, units, dtype)
    assert is_duck_dask_array(encoded_times)
    assert encoded_times.chunks == times.chunks
    if units is not None and dtype is not None:
        assert encoding_units == units
        assert encoded_times.dtype == dtype
    else:
        assert encoding_units == 'nanoseconds'
        assert encoded_times.dtype == np.dtype('int64')
    decoded_times = decode_cf_timedelta(encoded_times, encoding_units)
    np.testing.assert_equal(decoded_times, times)