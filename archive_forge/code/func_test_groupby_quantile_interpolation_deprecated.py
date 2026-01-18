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
@pytest.mark.parametrize('as_dataset', [False, True])
def test_groupby_quantile_interpolation_deprecated(as_dataset: bool) -> None:
    array = xr.DataArray(data=[1, 2, 3, 4], coords={'x': [1, 1, 2, 2]}, dims='x')
    arr: xr.DataArray | xr.Dataset
    arr = array.to_dataset(name='name') if as_dataset else array
    with pytest.warns(FutureWarning, match='`interpolation` argument to quantile was renamed to `method`'):
        actual = arr.quantile(0.5, interpolation='lower')
    expected = arr.quantile(0.5, method='lower')
    assert_identical(actual, expected)
    with warnings.catch_warnings(record=True):
        with pytest.raises(TypeError, match='interpolation and method keywords'):
            arr.quantile(0.5, method='lower', interpolation='lower')