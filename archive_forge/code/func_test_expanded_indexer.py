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
def test_expanded_indexer(self) -> None:
    x = np.random.randn(10, 11, 12, 13, 14)
    y = np.arange(5)
    arr = ReturnItem()
    for i in [arr[:], arr[...], arr[0, :, 10], arr[..., 10], arr[:5, ..., 0], arr[..., 0, :], arr[y], arr[y, y], arr[..., y, y], arr[..., 0, 1, 2, 3, 4]]:
        j = indexing.expanded_indexer(i, x.ndim)
        assert_array_equal(x[i], x[j])
        assert_array_equal(self.set_to_zero(x, i), self.set_to_zero(x, j))
    with pytest.raises(IndexError, match='too many indices'):
        indexing.expanded_indexer(arr[1, 2, 3], 2)