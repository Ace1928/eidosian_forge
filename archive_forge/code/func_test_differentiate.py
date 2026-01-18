from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@pytest.mark.parametrize('dask', [True, False])
@pytest.mark.parametrize('edge_order', [1, 2])
def test_differentiate(dask, edge_order) -> None:
    rs = np.random.RandomState(42)
    coord = [0.2, 0.35, 0.4, 0.6, 0.7, 0.75, 0.76, 0.8]
    da = xr.DataArray(rs.randn(8, 6), dims=['x', 'y'], coords={'x': coord, 'z': 3, 'x2d': (('x', 'y'), rs.randn(8, 6))})
    if dask and has_dask:
        da = da.chunk({'x': 4})
    ds = xr.Dataset({'var': da})
    actual = da.differentiate('x', edge_order)
    expected_x = xr.DataArray(np.gradient(da, da['x'], axis=0, edge_order=edge_order), dims=da.dims, coords=da.coords)
    assert_equal(expected_x, actual)
    assert_equal(ds['var'].differentiate('x', edge_order=edge_order), ds.differentiate('x', edge_order=edge_order)['var'])
    assert_equal(da['x'], actual['x'])
    actual = da.differentiate('y', edge_order)
    expected_y = xr.DataArray(np.gradient(da, da['y'], axis=1, edge_order=edge_order), dims=da.dims, coords=da.coords)
    assert_equal(expected_y, actual)
    assert_equal(actual, ds.differentiate('y', edge_order=edge_order)['var'])
    assert_equal(ds['var'].differentiate('y', edge_order=edge_order), ds.differentiate('y', edge_order=edge_order)['var'])
    with pytest.raises(ValueError):
        da.differentiate('x2d')