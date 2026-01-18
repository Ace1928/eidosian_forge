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
@pytest.mark.parametrize(('freq', 'units'), FREQUENCIES_TO_ENCODING_UNITS.items())
def test_infer_datetime_units(freq, units) -> None:
    dates = pd.date_range('2000', periods=2, freq=freq)
    expected = f'{units} since 2000-01-01 00:00:00'
    assert expected == coding.times.infer_datetime_units(dates)