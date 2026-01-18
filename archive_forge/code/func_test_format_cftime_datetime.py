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
@pytest.mark.parametrize(['date_args', 'expected'], [((1, 2, 3, 4, 5, 6), '0001-02-03 04:05:06.000000'), ((10, 2, 3, 4, 5, 6), '0010-02-03 04:05:06.000000'), ((100, 2, 3, 4, 5, 6), '0100-02-03 04:05:06.000000'), ((1000, 2, 3, 4, 5, 6), '1000-02-03 04:05:06.000000')])
def test_format_cftime_datetime(date_args, expected) -> None:
    date_types = _all_cftime_date_types()
    for date_type in date_types.values():
        result = coding.times.format_cftime_datetime(date_type(*date_args))
        assert result == expected