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
def test_concat_constant_index(self):
    ds1 = Dataset({'foo': 1.5}, {'y': 1})
    ds2 = Dataset({'foo': 2.5}, {'y': 1})
    expected = Dataset({'foo': ('y', [1.5, 2.5]), 'y': [1, 1]})
    for mode in ['different', 'all', ['foo']]:
        actual = concat([ds1, ds2], 'y', data_vars=mode)
        assert_identical(expected, actual)
    with pytest.raises(merge.MergeError, match='conflicting values'):
        concat([ds1, ds2], 'new_dim', data_vars='minimal')