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
def test_implicit_indexing_adapter_copy_on_write() -> None:
    array = np.arange(10, dtype=np.int64)
    implicit = indexing.ImplicitToExplicitIndexingAdapter(indexing.CopyOnWriteArray(array))
    assert isinstance(implicit[:], indexing.ImplicitToExplicitIndexingAdapter)