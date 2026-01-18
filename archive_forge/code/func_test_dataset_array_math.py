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
def test_dataset_array_math(self) -> None:
    ds = self.make_example_math_dataset()
    expected = ds.map(lambda x: x - ds['foo'])
    assert_identical(expected, ds - ds['foo'])
    assert_identical(expected, -ds['foo'] + ds)
    assert_identical(expected, ds - ds['foo'].variable)
    assert_identical(expected, -ds['foo'].variable + ds)
    actual = ds.copy(deep=True)
    actual -= ds['foo']
    assert_identical(expected, actual)
    expected = ds.map(lambda x: x + ds['bar'])
    assert_identical(expected, ds + ds['bar'])
    actual = ds.copy(deep=True)
    actual += ds['bar']
    assert_identical(expected, actual)
    expected = Dataset({'bar': ds['bar'] + np.arange(3)})
    assert_identical(expected, ds[['bar']] + np.arange(3))
    assert_identical(expected, np.arange(3) + ds[['bar']])