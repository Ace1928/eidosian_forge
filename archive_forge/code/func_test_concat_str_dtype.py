from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
@pytest.mark.parametrize('dtype', [str, bytes])
def test_concat_str_dtype(self, dtype) -> None:
    a = PandasIndex(np.array(['a'], dtype=dtype), 'x', coord_dtype=dtype)
    b = PandasIndex(np.array(['b'], dtype=dtype), 'x', coord_dtype=dtype)
    expected = PandasIndex(np.array(['a', 'b'], dtype=dtype), 'x', coord_dtype=dtype)
    actual = PandasIndex.concat([a, b], 'x')
    assert actual.equals(expected)
    assert np.issubdtype(actual.coord_dtype, dtype)