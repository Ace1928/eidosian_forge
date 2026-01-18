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
def test_groupby_da_datetime() -> None:
    times = pd.date_range('2000-01-01', periods=4)
    foo = xr.DataArray([1, 2, 3, 4], coords=dict(time=times), dims='time')
    dd = times.to_pydatetime()
    reference_dates = [dd[0], dd[2]]
    labels = reference_dates[0:1] * 2 + reference_dates[1:2] * 2
    ind = xr.DataArray(labels, coords=dict(time=times), dims='time', name='reference_date')
    g = foo.groupby(ind)
    actual = g.sum(dim='time')
    expected = xr.DataArray([3, 7], coords=dict(reference_date=reference_dates), dims='reference_date')
    assert_equal(expected, actual)