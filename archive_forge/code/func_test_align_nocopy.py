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
def test_align_nocopy(self) -> None:
    x = Dataset({'foo': DataArray([1, 2, 3], coords=[('x', [1, 2, 3])])})
    y = Dataset({'foo': DataArray([1, 2], coords=[('x', [1, 2])])})
    expected_x2 = x
    expected_y2 = Dataset({'foo': DataArray([1, 2, np.nan], coords=[('x', [1, 2, 3])])})
    x2, y2 = align(x, y, copy=False, join='outer')
    assert_identical(expected_x2, x2)
    assert_identical(expected_y2, y2)
    assert source_ndarray(x['foo'].data) is source_ndarray(x2['foo'].data)
    x2, y2 = align(x, y, copy=True, join='outer')
    assert source_ndarray(x['foo'].data) is not source_ndarray(x2['foo'].data)
    assert_identical(expected_x2, x2)
    assert_identical(expected_y2, y2)