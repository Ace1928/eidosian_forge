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
def test_groupby_first_and_last(self) -> None:
    array = DataArray([1, 2, 3, 4, 5], dims='x')
    by = DataArray(['a'] * 2 + ['b'] * 3, dims='x', name='ab')
    expected = DataArray([1, 3], [('ab', ['a', 'b'])])
    actual = array.groupby(by).first()
    assert_identical(expected, actual)
    expected = DataArray([2, 5], [('ab', ['a', 'b'])])
    actual = array.groupby(by).last()
    assert_identical(expected, actual)
    array = DataArray(np.random.randn(5, 3), dims=['x', 'y'])
    expected = DataArray(array[[0, 2]], {'ab': ['a', 'b']}, ['ab', 'y'])
    actual = array.groupby(by).first()
    assert_identical(expected, actual)
    actual = array.groupby('x').first()
    expected = array
    assert_identical(expected, actual)