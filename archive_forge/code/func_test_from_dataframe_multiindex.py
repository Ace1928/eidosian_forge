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
def test_from_dataframe_multiindex(self) -> None:
    index = pd.MultiIndex.from_product([['a', 'b'], [1, 2, 3]], names=['x', 'y'])
    df = pd.DataFrame({'z': np.arange(6)}, index=index)
    expected = Dataset({'z': (('x', 'y'), [[0, 1, 2], [3, 4, 5]])}, coords={'x': ['a', 'b'], 'y': [1, 2, 3]})
    actual = Dataset.from_dataframe(df)
    assert_identical(actual, expected)
    df2 = df.iloc[[3, 2, 1, 0, 4, 5], :]
    actual = Dataset.from_dataframe(df2)
    assert_identical(actual, expected)
    df3 = df.iloc[:4, :]
    expected3 = Dataset({'z': (('x', 'y'), [[0, 1, 2], [3, np.nan, np.nan]])}, coords={'x': ['a', 'b'], 'y': [1, 2, 3]})
    actual = Dataset.from_dataframe(df3)
    assert_identical(actual, expected3)
    df_nonunique = df.iloc[[0, 0], :]
    with pytest.raises(ValueError, match='non-unique MultiIndex'):
        Dataset.from_dataframe(df_nonunique)