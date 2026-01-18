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
def test_dataset_diff_n2(self) -> None:
    ds = create_test_data(seed=1)
    actual = ds.diff('dim2', n=2)
    expected_dict = {}
    expected_dict['var1'] = DataArray(np.diff(ds['var1'].values, axis=1, n=2), {'dim2': ds['dim2'].values[2:]}, ['dim1', 'dim2'])
    expected_dict['var2'] = DataArray(np.diff(ds['var2'].values, axis=1, n=2), {'dim2': ds['dim2'].values[2:]}, ['dim1', 'dim2'])
    expected_dict['var3'] = ds['var3']
    expected = Dataset(expected_dict, coords={'time': ds['time'].values})
    expected.coords['numbers'] = ('dim3', ds['numbers'].values)
    assert_equal(expected, actual)