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
def test_groupby_bins_multidim(self) -> None:
    array = self.make_groupby_multidim_example_array()
    bins = [0, 15, 20]
    bin_coords = pd.cut(array['lat'].values.flat, bins).categories
    expected = DataArray([16, 40], dims='lat_bins', coords={'lat_bins': bin_coords})
    actual = array.groupby_bins('lat', bins).map(lambda x: x.sum())
    assert_identical(expected, actual)
    array['lat'].data = np.array([[10.0, 20.0], [20.0, 10.0]])
    expected = DataArray([28, 28], dims='lat_bins', coords={'lat_bins': bin_coords})
    actual = array.groupby_bins('lat', bins).map(lambda x: x.sum())
    assert_identical(expected, actual)
    bins = [-2, -1, 0, 1, 2]
    field = DataArray(np.ones((5, 3)), dims=('x', 'y'))
    by = DataArray(np.array([[-1.5, -1.5, 0.5, 1.5, 1.5] * 3]).reshape(5, 3), dims=('x', 'y'))
    actual = field.groupby_bins(by, bins=bins).count()
    bincoord = np.array([pd.Interval(left, right, closed='right') for left, right in zip(bins[:-1], bins[1:])], dtype=object)
    expected = DataArray(np.array([6, np.nan, 3, 6]), dims='group_bins', coords={'group_bins': bincoord})
    assert_identical(actual, expected)