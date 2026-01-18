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
def test_groupby_math(self) -> None:
    array = self.da
    grouped = array.groupby('abc')
    expected_agg = (grouped.mean(...) - np.arange(3)).rename(None)
    actual = grouped - DataArray(range(3), [('abc', ['a', 'b', 'c'])])
    actual_agg = actual.groupby('abc').mean(...)
    assert_allclose(expected_agg, actual_agg)
    with pytest.raises(TypeError, match='only support binary ops'):
        grouped + 1
    with pytest.raises(TypeError, match='only support binary ops'):
        grouped + grouped
    with pytest.raises(TypeError, match='in-place operations'):
        array += grouped