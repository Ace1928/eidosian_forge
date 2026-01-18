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
@pytest.mark.parametrize('orig_coords', [{}, {'x': range(4)}])
def test_assign_coords_new_multiindex(self, orig_coords) -> None:
    ds = Dataset(coords=orig_coords)
    midx = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [0, 1, 0, 1]], names=('one', 'two'))
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    expected = Dataset(coords=midx_coords)
    with pytest.warns(FutureWarning, match='.*`pandas.MultiIndex`.*no longer be implicitly promoted.*'):
        actual = ds.assign_coords({'x': midx})
    assert_identical(actual, expected)
    actual = ds.assign_coords(midx_coords)
    assert_identical(actual, expected)