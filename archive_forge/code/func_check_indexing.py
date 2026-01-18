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
def check_indexing(v_eager, v_lazy, indexers):
    for indexer in indexers:
        actual = v_lazy[indexer]
        expected = v_eager[indexer]
        assert expected.shape == actual.shape
        assert isinstance(actual._data, (indexing.LazilyVectorizedIndexedArray, indexing.LazilyIndexedArray))
        assert_array_equal(expected, actual)
        v_eager = expected
        v_lazy = actual