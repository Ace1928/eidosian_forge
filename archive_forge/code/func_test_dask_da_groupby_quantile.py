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
@requires_dask
def test_dask_da_groupby_quantile() -> None:
    expected = xr.DataArray(data=[2, 5], coords={'x': [1, 2], 'quantile': 0.5}, dims='x')
    array = xr.DataArray(data=[1, 2, 3, 4, 5, 6], coords={'x': [1, 1, 1, 2, 2, 2]}, dims='x')
    with pytest.raises(ValueError):
        array.chunk(x=1).groupby('x').quantile(0.5)
    actual = array.chunk(x=3).groupby('x').quantile(0.5)
    assert_identical(expected, actual)
    actual = array.chunk(x=-1).groupby('x').quantile(0.5)
    assert_identical(expected, actual)