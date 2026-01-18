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
def test_lazily_indexed_array_vindex_setitem(self) -> None:
    lazy = indexing.LazilyIndexedArray(np.random.rand(10, 20, 30))
    indexer = indexing.VectorizedIndexer((np.array([0, 1]), np.array([0, 1]), slice(None, None, None)))
    with pytest.raises(NotImplementedError, match='Lazy item assignment with the vectorized indexer is not yet'):
        lazy.vindex[indexer] = 0