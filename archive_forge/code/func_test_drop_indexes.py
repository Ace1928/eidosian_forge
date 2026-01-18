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
def test_drop_indexes(self) -> None:
    ds = Dataset(coords={'x': ('x', [0, 1, 2]), 'y': ('y', [3, 4, 5]), 'foo': ('x', ['a', 'a', 'b'])})
    actual = ds.drop_indexes('x')
    assert 'x' not in actual.xindexes
    assert type(actual.x.variable) is Variable
    actual = ds.drop_indexes(['x', 'y'])
    assert 'x' not in actual.xindexes
    assert 'y' not in actual.xindexes
    assert type(actual.x.variable) is Variable
    assert type(actual.y.variable) is Variable
    with pytest.raises(ValueError, match="The coordinates \\('not_a_coord',\\) are not found in the dataset coordinates"):
        ds.drop_indexes('not_a_coord')
    with pytest.raises(ValueError, match='those coordinates do not have an index'):
        ds.drop_indexes('foo')
    actual = ds.drop_indexes(['foo', 'not_a_coord'], errors='ignore')
    assert_identical(actual, ds)
    midx = pd.MultiIndex.from_tuples([[1, 2], [3, 4]], names=['a', 'b'])
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    ds = Dataset(coords=midx_coords)
    with pytest.raises(ValueError, match='.*would corrupt the following index.*'):
        ds.drop_indexes('a')