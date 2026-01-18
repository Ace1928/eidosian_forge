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
def test_expand_dims_mixed_int_and_coords(self) -> None:
    original = Dataset({'x': ('a', np.random.randn(3)), 'y': (['b', 'a'], np.random.randn(4, 3))}, coords={'a': np.linspace(0, 1, 3), 'b': np.linspace(0, 1, 4), 'c': np.linspace(0, 1, 5)})
    actual = original.expand_dims({'d': 4, 'e': ['l', 'm', 'n']})
    expected = Dataset({'x': xr.DataArray(original['x'].values * np.ones([4, 3, 3]), coords=dict(d=range(4), e=['l', 'm', 'n'], a=np.linspace(0, 1, 3)), dims=['d', 'e', 'a']).drop_vars('d'), 'y': xr.DataArray(original['y'].values * np.ones([4, 3, 4, 3]), coords=dict(d=range(4), e=['l', 'm', 'n'], b=np.linspace(0, 1, 4), a=np.linspace(0, 1, 3)), dims=['d', 'e', 'b', 'a']).drop_vars('d')}, coords={'c': np.linspace(0, 1, 5)})
    assert_identical(actual, expected)