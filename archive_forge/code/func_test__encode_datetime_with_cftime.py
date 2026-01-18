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
def test__encode_datetime_with_cftime() -> None:
    import cftime
    calendar = 'gregorian'
    times = cftime.num2date([0, 1], 'hours since 2000-01-01', calendar)
    encoding_units = 'days since 2000-01-01'
    try:
        expected = cftime.date2num(times, encoding_units, calendar, longdouble=False)
    except TypeError:
        expected = cftime.date2num(times, encoding_units, calendar)
    result = _encode_datetime_with_cftime(times, encoding_units, calendar)
    np.testing.assert_equal(result, expected)