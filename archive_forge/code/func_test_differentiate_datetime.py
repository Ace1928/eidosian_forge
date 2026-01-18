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
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
@pytest.mark.parametrize('dask', [True, False])
def test_differentiate_datetime(dask) -> None:
    rs = np.random.RandomState(42)
    coord = np.array(['2004-07-13', '2006-01-13', '2010-08-13', '2010-09-13', '2010-10-11', '2010-12-13', '2011-02-13', '2012-08-13'], dtype='datetime64')
    da = xr.DataArray(rs.randn(8, 6), dims=['x', 'y'], coords={'x': coord, 'z': 3, 'x2d': (('x', 'y'), rs.randn(8, 6))})
    if dask and has_dask:
        da = da.chunk({'x': 4})
    actual = da.differentiate('x', edge_order=1, datetime_unit='D')
    expected_x = xr.DataArray(np.gradient(da, da['x'].variable._to_numeric(datetime_unit='D'), axis=0, edge_order=1), dims=da.dims, coords=da.coords)
    assert_equal(expected_x, actual)
    actual2 = da.differentiate('x', edge_order=1, datetime_unit='h')
    assert np.allclose(actual, actual2 * 24)
    actual = da['x'].differentiate('x', edge_order=1, datetime_unit='D')
    assert np.allclose(actual, 1.0)
    da = xr.DataArray(coord.astype('datetime64[ms]'), dims=['x'], coords={'x': coord})
    actual = da.differentiate('x', edge_order=1)
    assert np.allclose(actual, 1.0)