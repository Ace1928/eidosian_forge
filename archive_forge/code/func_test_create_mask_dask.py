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
def test_create_mask_dask() -> None:
    da = pytest.importorskip('dask.array')
    indexer = indexing.OuterIndexer((1, slice(2), np.array([0, -1, 2])))
    expected = np.array(2 * [[False, True, False]])
    actual = indexing.create_mask(indexer, (5, 5, 5), da.empty((2, 3), chunks=((1, 1), (2, 1))))
    assert actual.chunks == ((1, 1), (2, 1))
    np.testing.assert_array_equal(expected, actual)
    indexer_vec = indexing.VectorizedIndexer((np.array([0, -1, 2]), slice(None), np.array([0, 1, -1])))
    expected = np.array([[False, True, True]] * 2).T
    actual = indexing.create_mask(indexer_vec, (5, 2), da.empty((3, 2), chunks=((3,), (2,))))
    assert isinstance(actual, da.Array)
    np.testing.assert_array_equal(expected, actual)
    with pytest.raises(ValueError):
        indexing.create_mask(indexer_vec, (5, 2), da.empty((5,), chunks=(1,)))