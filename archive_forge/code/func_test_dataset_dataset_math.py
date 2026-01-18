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
def test_dataset_dataset_math(self) -> None:
    ds = self.make_example_math_dataset()
    assert_identical(ds, ds + 0 * ds)
    assert_identical(ds, ds + {'foo': 0, 'bar': 0})
    expected = ds.map(lambda x: 2 * x)
    assert_identical(expected, 2 * ds)
    assert_identical(expected, ds + ds)
    assert_identical(expected, ds + ds.data_vars)
    assert_identical(expected, ds + dict(ds.data_vars))
    actual = ds.copy(deep=True)
    expected_id = id(actual)
    actual += ds
    assert_identical(expected, actual)
    assert expected_id == id(actual)
    assert_identical(ds == ds, ds.notnull())
    subsampled = ds.isel(y=slice(2))
    expected = 2 * subsampled
    assert_identical(expected, subsampled + ds)
    assert_identical(expected, ds + subsampled)