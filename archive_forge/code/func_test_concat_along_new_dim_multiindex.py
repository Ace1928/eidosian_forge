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
def test_concat_along_new_dim_multiindex(self) -> None:
    level_names = ['x_level_0', 'x_level_1']
    midx = pd.MultiIndex.from_product([[1, 2, 3], ['a', 'b']], names=level_names)
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    ds = Dataset(coords=midx_coords)
    concatenated = concat([ds], 'new')
    actual = list(concatenated.xindexes.get_all_coords('x'))
    expected = ['x'] + level_names
    assert actual == expected