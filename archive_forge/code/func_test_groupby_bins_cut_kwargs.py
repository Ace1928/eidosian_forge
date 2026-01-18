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
@pytest.mark.parametrize('use_flox', [True, False])
def test_groupby_bins_cut_kwargs(use_flox: bool) -> None:
    da = xr.DataArray(np.arange(12).reshape(6, 2), dims=('x', 'y'))
    x_bins = (0, 2, 4, 6)
    with xr.set_options(use_flox=use_flox):
        actual = da.groupby_bins('x', bins=x_bins, include_lowest=True, right=False, squeeze=False).mean()
    expected = xr.DataArray(np.array([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]]), dims=('x_bins', 'y'), coords={'x_bins': ('x_bins', pd.IntervalIndex.from_breaks(x_bins, closed='left'))})
    assert_identical(expected, actual)