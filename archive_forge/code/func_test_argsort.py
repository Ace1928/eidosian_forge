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
def test_argsort(self, index):
    if isinstance(index, CategoricalIndex):
        return
    result = index.argsort()
    expected = np.array(index).argsort()
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)