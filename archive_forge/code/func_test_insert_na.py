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
def test_insert_na(self, nulls_fixture, simple_index):
    index = simple_index
    na_val = nulls_fixture
    if na_val is pd.NaT:
        expected = Index([index[0], pd.NaT] + list(index[1:]), dtype=object)
    else:
        expected = Index([index[0], np.nan] + list(index[1:]))
        if index.dtype.kind == 'f':
            expected = Index(expected, dtype=index.dtype)
    result = index.insert(1, na_val)
    tm.assert_index_equal(result, expected, exact=True)