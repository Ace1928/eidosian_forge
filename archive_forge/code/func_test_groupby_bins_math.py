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
@pytest.mark.parametrize('indexed_coord', [True, False])
def test_groupby_bins_math(indexed_coord) -> None:
    N = 7
    da = DataArray(np.random.random((N, N)), dims=('x', 'y'))
    if indexed_coord:
        da['x'] = np.arange(N)
        da['y'] = np.arange(N)
    g = da.groupby_bins('x', np.arange(0, N + 1, 3))
    mean = g.mean()
    expected = da.isel(x=slice(1, None)) - mean.isel(x_bins=('x', [0, 0, 0, 1, 1, 1]))
    actual = g - mean
    assert_identical(expected, actual)