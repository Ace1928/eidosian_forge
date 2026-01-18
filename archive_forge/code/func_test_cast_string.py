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
def test_cast_string(self, dtype):
    result = self._index_cls(['0', '1', '2'], dtype=dtype)
    expected = self._index_cls([0, 1, 2], dtype=dtype)
    tm.assert_index_equal(result, expected)