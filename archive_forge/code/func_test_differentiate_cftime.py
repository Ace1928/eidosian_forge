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
@pytest.mark.skipif(not has_cftime, reason='Test requires cftime.')
@pytest.mark.parametrize('dask', [True, False])
def test_differentiate_cftime(dask) -> None:
    rs = np.random.RandomState(42)
    coord = xr.cftime_range('2000', periods=8, freq='2ME')
    da = xr.DataArray(rs.randn(8, 6), coords={'time': coord, 'z': 3, 't2d': (('time', 'y'), rs.randn(8, 6))}, dims=['time', 'y'])
    if dask and has_dask:
        da = da.chunk({'time': 4})
    actual = da.differentiate('time', edge_order=1, datetime_unit='D')
    expected_data = np.gradient(da, da['time'].variable._to_numeric(datetime_unit='D'), axis=0, edge_order=1)
    expected = xr.DataArray(expected_data, coords=da.coords, dims=da.dims)
    assert_equal(expected, actual)
    actual2 = da.differentiate('time', edge_order=1, datetime_unit='h')
    assert_allclose(actual, actual2 * 24)
    actual = da['time'].differentiate('time', edge_order=1, datetime_unit='D')
    assert_allclose(actual, xr.ones_like(da['time']).astype(float))