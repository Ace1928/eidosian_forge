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
def test_concat_do_not_promote(self) -> None:
    objs = [Dataset({'y': ('t', [1])}, {'x': 1, 't': [0]}), Dataset({'y': ('t', [2])}, {'x': 1, 't': [0]})]
    expected = Dataset({'y': ('t', [1, 2])}, {'x': 1, 't': [0, 0]})
    actual = concat(objs, 't')
    assert_identical(expected, actual)
    objs = [Dataset({'y': ('t', [1])}, {'x': 1, 't': [0]}), Dataset({'y': ('t', [2])}, {'x': 2, 't': [0]})]
    with pytest.raises(ValueError):
        concat(objs, 't', coords='minimal')