from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('name', [None, 'foo'])
@pytest.mark.parametrize('ordered', [True, False])
def test_astype_category(self, copy, name, ordered, simple_index):
    idx = simple_index
    if name:
        idx = idx.rename(name)
    dtype = CategoricalDtype(ordered=ordered)
    result = idx.astype(dtype, copy=copy)
    expected = CategoricalIndex(idx, name=name, ordered=ordered)
    tm.assert_index_equal(result, expected, exact=True)
    dtype = CategoricalDtype(idx.unique().tolist()[:-1], ordered)
    result = idx.astype(dtype, copy=copy)
    expected = CategoricalIndex(idx, name=name, dtype=dtype)
    tm.assert_index_equal(result, expected, exact=True)
    if ordered is False:
        result = idx.astype('category', copy=copy)
        expected = CategoricalIndex(idx, name=name)
        tm.assert_index_equal(result, expected, exact=True)