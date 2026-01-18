from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.types import T_Xarray
from xarray.tests import (
def test_indexer(data: T_Xarray, x: Any, expected: indexing.IndexSelResult) -> None:
    results = indexing.map_index_queries(data, {'x': x})
    assert results.dim_indexers.keys() == expected.dim_indexers.keys()
    assert_array_equal(results.dim_indexers['x'], expected.dim_indexers['x'])
    assert results.indexes.keys() == expected.indexes.keys()
    for k in results.indexes:
        assert results.indexes[k].equals(expected.indexes[k])
    assert results.variables.keys() == expected.variables.keys()
    for k in results.variables:
        assert_array_equal(results.variables[k], expected.variables[k])
    assert set(results.drop_coords) == set(expected.drop_coords)
    assert set(results.drop_indexes) == set(expected.drop_indexes)
    assert results.rename_dims == expected.rename_dims