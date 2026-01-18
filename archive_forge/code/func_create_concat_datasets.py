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
def create_concat_datasets(num_datasets: int=2, seed: int | None=None, include_day: bool=True) -> list[Dataset]:
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal(size=(1, 4))
    lon = rng.standard_normal(size=(1, 4))
    result = []
    variables = ['temperature', 'pressure', 'humidity', 'precipitation', 'cloud_cover']
    for i in range(num_datasets):
        if include_day:
            data_tuple = (['x', 'y', 'day'], rng.standard_normal(size=(1, 4, 2)))
            data_vars = {v: data_tuple for v in variables}
            result.append(Dataset(data_vars=data_vars, coords={'lat': (['x', 'y'], lat), 'lon': (['x', 'y'], lon), 'day': ['day' + str(i * 2 + 1), 'day' + str(i * 2 + 2)]}))
        else:
            data_tuple = (['x', 'y'], rng.standard_normal(size=(1, 4)))
            data_vars = {v: data_tuple for v in variables}
            result.append(Dataset(data_vars=data_vars, coords={'lat': (['x', 'y'], lat), 'lon': (['x', 'y'], lon)}))
    return result