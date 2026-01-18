from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_cftime
def test_get_clean_interp_index_potential_overflow():
    da = xr.DataArray([0, 1, 2], dims=('time',), coords={'time': xr.cftime_range('0000-01-01', periods=3, calendar='360_day')})
    get_clean_interp_index(da, 'time')