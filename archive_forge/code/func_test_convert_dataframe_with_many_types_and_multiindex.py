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
def test_convert_dataframe_with_many_types_and_multiindex(self) -> None:
    df = pd.DataFrame({'a': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6).astype('u1'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True], 'f': pd.Categorical(list('abc')), 'g': pd.date_range('20130101', periods=3), 'h': pd.date_range('20130101', periods=3, tz='America/New_York')})
    df.index = pd.MultiIndex.from_product([['a'], range(3)], names=['one', 'two'])
    roundtripped = Dataset.from_dataframe(df).to_dataframe()
    expected = df.apply(np.asarray)
    assert roundtripped.equals(expected)