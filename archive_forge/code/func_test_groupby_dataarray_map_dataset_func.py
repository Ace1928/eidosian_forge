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
def test_groupby_dataarray_map_dataset_func() -> None:
    da = DataArray([1, 2, 3, 4], coords={'x': [0, 0, 1, 1]}, dims='x', name='foo')
    actual = da.groupby('x').map(lambda grp: grp.mean().to_dataset())
    expected = xr.Dataset({'foo': ('x', [1.5, 3.5])}, coords={'x': [0, 1]})
    assert_identical(actual, expected)