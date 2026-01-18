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
def test_create_mask_vectorized_indexer() -> None:
    indexer = indexing.VectorizedIndexer((np.array([0, -1, 2]), np.array([0, 1, -1])))
    expected = np.array([False, True, True])
    actual = indexing.create_mask(indexer, (5,))
    np.testing.assert_array_equal(expected, actual)
    indexer = indexing.VectorizedIndexer((np.array([0, -1, 2]), slice(None), np.array([0, 1, -1])))
    expected = np.array([[False, True, True]] * 2).T
    actual = indexing.create_mask(indexer, (5, 2))
    np.testing.assert_array_equal(expected, actual)