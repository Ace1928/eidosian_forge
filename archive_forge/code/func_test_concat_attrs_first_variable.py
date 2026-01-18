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
@pytest.mark.parametrize('attr1', ({'a': {'meta': [10, 20, 30]}}, {'a': [1, 2, 3]}, {}))
@pytest.mark.parametrize('attr2', ({'a': [1, 2, 3]}, {}))
def test_concat_attrs_first_variable(attr1, attr2) -> None:
    arrs = [DataArray([[1], [2]], dims=['x', 'y'], attrs=attr1), DataArray([[3], [4]], dims=['x', 'y'], attrs=attr2)]
    concat_attrs = concat(arrs, 'y').attrs
    assert concat_attrs == attr1