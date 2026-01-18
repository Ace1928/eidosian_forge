from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_from_integer_array(self):
    arr = np.array([1, 2, 3], dtype=np.int64)
    data = pd.array(arr, dtype='Int64')
    if self.array_cls is PeriodArray:
        expected = self.array_cls(arr, dtype=self.example_dtype)
        result = self.array_cls(data, dtype=self.example_dtype)
    else:
        expected = self.array_cls._from_sequence(arr, dtype=self.example_dtype)
        result = self.array_cls._from_sequence(data, dtype=self.example_dtype)
    tm.assert_extension_array_equal(result, expected)