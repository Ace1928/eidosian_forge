import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dot_1d_ndarray(self, obj, expected):
    row = obj.iloc[0] if obj.ndim == 2 else obj
    result = obj.dot(row.values)
    expected = obj.dot(row)
    self.reduced_dim_assert(result, expected)