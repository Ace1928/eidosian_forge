from datetime import (
import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_float64_index_difference(self):
    float_index = Index([1.0, 2, 3])
    string_index = Index(['1', '2', '3'])
    result = float_index.difference(string_index)
    tm.assert_index_equal(result, float_index)
    result = string_index.difference(float_index)
    tm.assert_index_equal(result, string_index)