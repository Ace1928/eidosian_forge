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
@pytest.mark.parametrize('method', ['sum', 'mean', 'median'])
def test_groupby_reductions(self, method) -> None:
    array = self.da
    grouped = array.groupby('abc')
    reduction = getattr(np, method)
    expected = Dataset({'foo': Variable(['x', 'abc'], np.array([reduction(self.x[:, :9], axis=-1), reduction(self.x[:, 10:], axis=-1), reduction(self.x[:, 9:10], axis=-1)]).T), 'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
    with xr.set_options(use_flox=False):
        actual_legacy = getattr(grouped, method)(dim='y')
    with xr.set_options(use_flox=True):
        actual_npg = getattr(grouped, method)(dim='y')
    assert_allclose(expected, actual_legacy)
    assert_allclose(expected, actual_npg)