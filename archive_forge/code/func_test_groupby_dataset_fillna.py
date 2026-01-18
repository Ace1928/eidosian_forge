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
def test_groupby_dataset_fillna() -> None:
    ds = Dataset({'a': ('x', [np.nan, 1, np.nan, 3])}, {'x': [0, 1, 2, 3]})
    expected = Dataset({'a': ('x', range(4))}, {'x': [0, 1, 2, 3]})
    for target in [ds, expected]:
        target.coords['b'] = ('x', [0, 0, 1, 1])
    actual = ds.groupby('b').fillna(DataArray([0, 2], dims='b'))
    assert_identical(expected, actual)
    actual = ds.groupby('b').fillna(Dataset({'a': ('b', [0, 2])}))
    assert_identical(expected, actual)
    ds.attrs['attr'] = 'ds'
    ds.a.attrs['attr'] = 'da'
    actual = ds.groupby('b').fillna(Dataset({'a': ('b', [0, 2])}))
    assert actual.attrs == ds.attrs
    assert actual.a.name == 'a'
    assert actual.a.attrs == ds.a.attrs