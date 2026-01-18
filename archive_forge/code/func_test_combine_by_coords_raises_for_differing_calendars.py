from __future__ import annotations
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from xarray import (
from xarray.core import dtypes
from xarray.core.combine import (
from xarray.tests import assert_equal, assert_identical, requires_cftime
from xarray.tests.test_dataset import create_test_data
@requires_cftime
def test_combine_by_coords_raises_for_differing_calendars():
    import cftime
    time_1 = [cftime.DatetimeGregorian(2000, 1, 1)]
    time_2 = [cftime.DatetimeProlepticGregorian(2001, 1, 1)]
    da_1 = DataArray([0], dims=['time'], coords=[time_1], name='a').to_dataset()
    da_2 = DataArray([1], dims=['time'], coords=[time_2], name='a').to_dataset()
    error_msg = "Cannot combine along dimension 'time' with mixed types. Found:.* If importing data directly from a file then setting `use_cftime=True` may fix this issue."
    with pytest.raises(TypeError, match=error_msg):
        combine_by_coords([da_1, da_2])