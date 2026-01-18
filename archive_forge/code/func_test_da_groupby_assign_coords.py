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
def test_da_groupby_assign_coords() -> None:
    actual = xr.DataArray([[3, 4, 5], [6, 7, 8]], dims=['y', 'x'], coords={'y': range(2), 'x': range(3)})
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        actual1 = actual.groupby('x').assign_coords({'y': [-1, -2]})
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        actual2 = actual.groupby('x').assign_coords(y=[-1, -2])
    expected = xr.DataArray([[3, 4, 5], [6, 7, 8]], dims=['y', 'x'], coords={'y': [-1, -2], 'x': range(3)})
    assert_identical(expected, actual1)
    assert_identical(expected, actual2)