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
def test_groupby_count(self) -> None:
    array = DataArray([0, 0, np.nan, np.nan, 0, 0], coords={'cat': ('x', ['a', 'b', 'b', 'c', 'c', 'c'])}, dims='x')
    actual = array.groupby('cat').count()
    expected = DataArray([1, 1, 2], coords=[('cat', ['a', 'b', 'c'])])
    assert_identical(actual, expected)