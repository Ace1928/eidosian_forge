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
def test_categorical_index(self) -> None:
    cat = pd.CategoricalIndex(['foo', 'bar', 'foo'], categories=['foo', 'bar', 'baz', 'qux', 'quux', 'corge'])
    ds = xr.Dataset({'var': ('cat', np.arange(3))}, coords={'cat': ('cat', cat), 'c': ('cat', [0, 1, 1])})
    actual1 = ds.sel(cat='foo')
    expected1 = ds.isel(cat=[0, 2])
    assert_identical(expected1, actual1)
    actual2 = ds.sel(cat='foo')['cat'].values
    assert (actual2 == np.array(['foo', 'foo'])).all()
    ds = ds.set_index(index=['cat', 'c'])
    actual3 = ds.unstack('index')
    assert actual3['var'].shape == (2, 2)