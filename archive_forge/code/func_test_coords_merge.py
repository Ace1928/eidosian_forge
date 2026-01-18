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
def test_coords_merge(self) -> None:
    orig_coords = Dataset(coords={'a': ('x', [1, 2]), 'x': [0, 1]}).coords
    other_coords = Dataset(coords={'b': ('x', ['a', 'b']), 'x': [0, 1]}).coords
    expected = Dataset(coords={'a': ('x', [1, 2]), 'b': ('x', ['a', 'b']), 'x': [0, 1]})
    actual = orig_coords.merge(other_coords)
    assert_identical(expected, actual)
    actual = other_coords.merge(orig_coords)
    assert_identical(expected, actual)
    other_coords = Dataset(coords={'x': ('x', ['a'])}).coords
    with pytest.raises(MergeError):
        orig_coords.merge(other_coords)
    other_coords = Dataset(coords={'x': ('x', ['a', 'b'])}).coords
    with pytest.raises(MergeError):
        orig_coords.merge(other_coords)
    other_coords = Dataset(coords={'x': ('x', ['a', 'b', 'c'])}).coords
    with pytest.raises(MergeError):
        orig_coords.merge(other_coords)
    other_coords = Dataset(coords={'a': ('x', [8, 9])}).coords
    expected = Dataset(coords={'x': range(2)})
    actual = orig_coords.merge(other_coords)
    assert_identical(expected, actual)
    actual = other_coords.merge(orig_coords)
    assert_identical(expected, actual)
    other_coords = Dataset(coords={'x': np.nan}).coords
    actual = orig_coords.merge(other_coords)
    assert_identical(orig_coords.to_dataset(), actual)
    actual = other_coords.merge(orig_coords)
    assert_identical(orig_coords.to_dataset(), actual)