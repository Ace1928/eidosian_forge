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
def test_concat_combine_attrs_kwarg(self) -> None:
    da1 = DataArray([0], coords=[('x', [0])], attrs={'b': 42})
    da2 = DataArray([0], coords=[('x', [1])], attrs={'b': 42, 'c': 43})
    expected: dict[CombineAttrsOptions, Any] = {}
    expected['drop'] = DataArray([0, 0], coords=[('x', [0, 1])])
    expected['no_conflicts'] = DataArray([0, 0], coords=[('x', [0, 1])], attrs={'b': 42, 'c': 43})
    expected['override'] = DataArray([0, 0], coords=[('x', [0, 1])], attrs={'b': 42})
    with pytest.raises(ValueError, match="combine_attrs='identical'"):
        actual = concat([da1, da2], dim='x', combine_attrs='identical')
    with pytest.raises(ValueError, match="combine_attrs='no_conflicts'"):
        da3 = da2.copy(deep=True)
        da3.attrs['b'] = 44
        actual = concat([da1, da3], dim='x', combine_attrs='no_conflicts')
    for combine_attrs in expected:
        actual = concat([da1, da2], dim='x', combine_attrs=combine_attrs)
        assert_identical(actual, expected[combine_attrs])