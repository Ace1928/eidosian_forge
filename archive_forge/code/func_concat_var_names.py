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
@pytest.fixture
def concat_var_names() -> Callable:

    def get_varnames(var_cnt: int=10, list_cnt: int=10) -> list[list[str]]:
        orig = [f'd{i:02d}' for i in range(var_cnt)]
        var_names = []
        for i in range(0, list_cnt):
            l1 = orig.copy()
            var_names.append(l1)
        return var_names
    return get_varnames