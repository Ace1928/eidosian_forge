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
@pytest.mark.parametrize('obj', [xr.DataArray([1, 2, 3, 4, 5, 6], [('x', [1, 1, 1, 2, 2, 2])]), xr.Dataset({'foo': ('x', [1, 2, 3, 4, 5, 6])}, {'x': [1, 1, 1, 2, 2, 2]})])
def test_groupby_map_shrink_groups(obj) -> None:
    expected = obj.isel(x=[0, 1, 3, 4])
    actual = obj.groupby('x').map(lambda f: f.isel(x=[0, 1]))
    assert_identical(expected, actual)