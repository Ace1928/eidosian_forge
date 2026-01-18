from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@pytest.mark.parametrize('dim', [True, False])
@pytest.mark.parametrize('coord', [True, False])
def test_concat_fill_missing_variables(concat_var_names, create_concat_ds, dim: bool, coord: bool) -> None:
    var_names = concat_var_names()
    drop_idx = [0, 7, 6, 4, 4, 8, 0, 6, 2, 0]
    expected = concat(create_concat_ds(var_names, dim=dim, coord=coord), dim='time', data_vars='all')
    for i, idx in enumerate(drop_idx):
        if dim:
            expected[var_names[0][idx]][i * 2:i * 2 + 2] = np.nan
        else:
            expected[var_names[0][idx]][i] = np.nan
    concat_ds = create_concat_ds(var_names, dim=dim, coord=coord, drop_idx=drop_idx)
    actual = concat(concat_ds, dim='time', data_vars='all')
    assert list(actual.data_vars.keys()) == ['d01', 'd02', 'd03', 'd04', 'd05', 'd06', 'd07', 'd08', 'd09', 'd00']
    assert_identical(actual, expected)