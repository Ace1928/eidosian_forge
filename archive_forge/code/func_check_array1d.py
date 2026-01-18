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
def check_array1d(indexer_cls):
    value, = indexer_cls((np.arange(3, dtype=np.int32),)).tuple
    assert value.dtype == np.int64
    np.testing.assert_array_equal(value, [0, 1, 2])