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
def test_basic_indexer() -> None:
    check_integer(indexing.BasicIndexer)
    check_slice(indexing.BasicIndexer)
    with pytest.raises(TypeError):
        check_array1d(indexing.BasicIndexer)
    with pytest.raises(TypeError):
        check_array2d(indexing.BasicIndexer)