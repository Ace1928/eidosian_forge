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
def test_insert_non_na(self, simple_index):
    index = simple_index
    result = index.insert(0, index[0])
    expected = Index([index[0]] + list(index), dtype=index.dtype)
    tm.assert_index_equal(result, expected, exact=True)