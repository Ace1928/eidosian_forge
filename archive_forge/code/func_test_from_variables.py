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
def test_from_variables(self) -> None:
    v_level1 = xr.Variable('x', [1, 2, 3], attrs={'unit': 'm'}, encoding={'dtype': np.int32})
    v_level2 = xr.Variable('x', ['a', 'b', 'c'], attrs={'unit': 'm'}, encoding={'dtype': 'U'})
    index = PandasMultiIndex.from_variables({'level1': v_level1, 'level2': v_level2}, options={})
    expected_idx = pd.MultiIndex.from_arrays([v_level1.data, v_level2.data])
    assert index.dim == 'x'
    assert index.index.equals(expected_idx)
    assert index.index.name == 'x'
    assert list(index.index.names) == ['level1', 'level2']
    var = xr.Variable(('x', 'y'), [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match='.*only accepts 1-dimensional variables.*'):
        PandasMultiIndex.from_variables({'var': var}, options={})
    v_level3 = xr.Variable('y', [4, 5, 6])
    with pytest.raises(ValueError, match='unmatched dimensions for multi-index variables.*'):
        PandasMultiIndex.from_variables({'level1': v_level1, 'level3': v_level3}, options={})