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
@pytest.mark.parametrize('indexer_class, key, value', [(indexing.OuterIndexer, (0, 1, slice(None, None, None)), 10), (indexing.BasicIndexer, (0, 1, slice(None, None, None)), 10)])
def test_lazily_indexed_array_setitem(self, indexer_class, key, value) -> None:
    original = np.random.rand(10, 20, 30)
    x = indexing.NumpyIndexingAdapter(original)
    lazy = indexing.LazilyIndexedArray(x)
    if indexer_class is indexing.BasicIndexer:
        indexer = indexer_class(key)
        lazy[indexer] = value
    elif indexer_class is indexing.OuterIndexer:
        indexer = indexer_class(key)
        lazy.oindex[indexer] = value
    assert_array_equal(original[key], value)