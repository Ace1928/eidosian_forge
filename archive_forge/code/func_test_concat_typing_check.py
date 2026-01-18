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
def test_concat_typing_check() -> None:
    ds = Dataset({'foo': 1}, {'bar': 2})
    da = Dataset({'foo': 3}, {'bar': 4}).to_dataarray(dim='foo')
    with pytest.raises(TypeError, match="The elements in the input list need to be either all 'Dataset's or all 'DataArray's"):
        concat([ds, da], dim='foo')
    with pytest.raises(TypeError, match="The elements in the input list need to be either all 'Dataset's or all 'DataArray's"):
        concat([da, ds], dim='foo')