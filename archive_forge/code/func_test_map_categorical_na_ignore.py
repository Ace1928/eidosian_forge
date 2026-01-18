from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_action, expected', [(None, Series([10.0, 42.0, np.nan])), ('ignore', Series([10, np.nan, np.nan]))])
def test_map_categorical_na_ignore(na_action, expected):
    values = pd.Categorical([1, np.nan, 2], categories=[10, 1, 2])
    ser = Series(values)
    result = ser.map({1: 10, np.nan: 42}, na_action=na_action)
    tm.assert_series_equal(result, expected)