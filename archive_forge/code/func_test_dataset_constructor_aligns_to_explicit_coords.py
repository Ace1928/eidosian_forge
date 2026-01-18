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
@pytest.mark.parametrize('unaligned_coords', ({'x': [2, 1, 0]}, {'x': (['x'], np.asarray([2, 1, 0]))}, {'x': (['x'], np.asarray([1, 2, 0]))}, {'x': pd.Index([2, 1, 0])}, {'x': Variable(dims='x', data=[0, 2, 1])}, {'x': IndexVariable(dims='x', data=[0, 1, 2])}, {'y': 42}, {'y': ('x', [2, 1, 0])}, {'y': ('x', np.asarray([2, 1, 0]))}, {'y': (['x'], np.asarray([2, 1, 0]))}))
@pytest.mark.parametrize('coords', ({'x': ('x', [0, 1, 2])}, {'x': [0, 1, 2]}))
def test_dataset_constructor_aligns_to_explicit_coords(unaligned_coords, coords) -> None:
    a = xr.DataArray([1, 2, 3], dims=['x'], coords=unaligned_coords)
    expected = xr.Dataset(coords=coords)
    expected['a'] = a
    result = xr.Dataset({'a': a}, coords=coords)
    assert_equal(expected, result)