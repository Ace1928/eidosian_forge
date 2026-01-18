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
def test_decode_abbreviation() -> None:
    """Test making sure we properly fall back to cftime on abbreviated units."""
    import cftime
    val = np.array([1586628000000.0])
    units = 'msecs since 1970-01-01T00:00:00Z'
    actual = coding.times.decode_cf_datetime(val, units)
    expected = coding.times.cftime_to_nptime(cftime.num2date(val, units))
    assert_array_equal(actual, expected)