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
def test_groupby_dataset_nan() -> None:
    ds = Dataset({'foo': ('x', [1, 2, 3, 4])}, {'bar': ('x', [1, 1, 2, np.nan])})
    actual = ds.groupby('bar').mean(...)
    expected = Dataset({'foo': ('bar', [1.5, 3]), 'bar': [1, 2]})
    assert_identical(actual, expected)