from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@requires_scipy
@pytest.mark.parametrize('x, expect_same_attrs', [(2.5, True), (np.array([2.5, 5]), True), (('x', np.array([0, 0.5, 1, 2]), dict(unit='s')), False)])
def test_coord_attrs(x, expect_same_attrs: bool) -> None:
    base_attrs = dict(foo='bar')
    ds = xr.Dataset(data_vars=dict(a=2 * np.arange(5)), coords={'x': ('x', np.arange(5), base_attrs)})
    has_same_attrs = ds.interp(x=x).x.attrs == base_attrs
    assert expect_same_attrs == has_same_attrs