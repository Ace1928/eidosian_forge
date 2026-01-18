from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
def test_concat_cftimeindex(date_type):
    da1 = xr.DataArray([1.0, 2.0], coords=[[date_type(1, 1, 1), date_type(1, 2, 1)]], dims=['time'])
    da2 = xr.DataArray([3.0, 4.0], coords=[[date_type(1, 3, 1), date_type(1, 4, 1)]], dims=['time'])
    da = xr.concat([da1, da2], dim='time')
    assert isinstance(da.xindexes['time'].to_pandas_index(), CFTimeIndex)