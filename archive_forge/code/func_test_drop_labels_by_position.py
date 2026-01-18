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
def test_drop_labels_by_position(self) -> None:
    data = Dataset({'A': (['x', 'y'], np.random.randn(2, 6)), 'x': ['a', 'b'], 'y': range(6)})
    assert len(data.coords['x']) == 2
    actual = data.drop_isel(x=0)
    expected = data.drop_sel(x='a')
    assert_identical(expected, actual)
    actual = data.drop_isel(x=[0])
    expected = data.drop_sel(x=['a'])
    assert_identical(expected, actual)
    actual = data.drop_isel(x=[0, 1])
    expected = data.drop_sel(x=['a', 'b'])
    assert_identical(expected, actual)
    assert actual.coords['x'].size == 0
    actual = data.drop_isel(x=[0, 1], y=range(0, 6, 2))
    expected = data.drop_sel(x=['a', 'b'], y=range(0, 6, 2))
    assert_identical(expected, actual)
    assert actual.coords['x'].size == 0
    with pytest.raises(KeyError):
        data.drop_isel(z=1)