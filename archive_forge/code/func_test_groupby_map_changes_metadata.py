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
def test_groupby_map_changes_metadata(self) -> None:

    def change_metadata(x):
        x.coords['x'] = x.coords['x'] * 2
        x.attrs['fruit'] = 'lemon'
        return x
    array = self.da
    grouped = array.groupby('abc')
    actual = grouped.map(change_metadata)
    expected = array.copy()
    expected = change_metadata(expected)
    assert_equal(expected, actual)