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
def test_constructor_with_coords(self) -> None:
    with pytest.raises(ValueError, match='found in both data_vars and'):
        Dataset({'a': ('x', [1])}, {'a': ('x', [1])})
    ds = Dataset({}, {'a': ('x', [1])})
    assert not ds.data_vars
    assert list(ds.coords.keys()) == ['a']
    mindex = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('level_1', 'level_2'))
    with pytest.raises(ValueError, match='conflicting MultiIndex'):
        with pytest.warns(FutureWarning, match='.*`pandas.MultiIndex`.*no longer be implicitly promoted.*'):
            Dataset({}, {'x': mindex, 'y': mindex})
            Dataset({}, {'x': mindex, 'level_1': range(4)})