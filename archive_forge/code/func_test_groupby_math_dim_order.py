from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
def test_groupby_math_dim_order() -> None:
    da = DataArray(np.ones((10, 10, 12)), dims=('x', 'y', 'time'), coords={'time': pd.date_range('2001-01-01', periods=12, freq='6h')})
    grouped = da.groupby('time.day')
    result = grouped - grouped.mean()
    assert result.dims == da.dims