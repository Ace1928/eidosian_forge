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
def test_encode_expected_failures() -> None:
    dates = pd.date_range('2000', periods=3)
    with pytest.raises(ValueError, match='invalid time units'):
        encode_cf_datetime(dates, units='days after 2000-01-01')
    with pytest.raises(ValueError, match='invalid reference date'):
        encode_cf_datetime(dates, units='days since NO_YEAR')