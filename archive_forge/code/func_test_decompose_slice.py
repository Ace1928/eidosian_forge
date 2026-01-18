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
@pytest.mark.parametrize('size', [100, 99])
@pytest.mark.parametrize('sl', [slice(1, -1, 1), slice(None, -1, 2), slice(-1, 1, -1), slice(-1, 1, -2)])
def test_decompose_slice(size, sl) -> None:
    x = np.arange(size)
    slice1, slice2 = indexing._decompose_slice(sl, size)
    expected = x[sl]
    actual = x[slice1][slice2]
    assert_array_equal(expected, actual)