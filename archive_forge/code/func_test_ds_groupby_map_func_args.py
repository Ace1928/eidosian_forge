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
def test_ds_groupby_map_func_args() -> None:

    def func(arg1, arg2, arg3=0):
        return arg1 + arg2 + arg3
    dataset = xr.Dataset({'foo': ('x', [1, 1, 1])}, {'x': [1, 2, 3]})
    expected = xr.Dataset({'foo': ('x', [3, 3, 3])}, {'x': [1, 2, 3]})
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        actual = dataset.groupby('x').map(func, args=(1,), arg3=1)
    assert_identical(expected, actual)