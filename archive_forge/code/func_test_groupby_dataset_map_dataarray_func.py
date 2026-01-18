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
def test_groupby_dataset_map_dataarray_func() -> None:
    ds = Dataset({'foo': ('x', [1, 2, 3, 4])}, coords={'x': [0, 0, 1, 1]})
    actual = ds.groupby('x').map(lambda grp: grp.foo.mean())
    expected = DataArray([1.5, 3.5], coords={'x': [0, 1]}, dims='x', name='foo')
    assert_identical(actual, expected)