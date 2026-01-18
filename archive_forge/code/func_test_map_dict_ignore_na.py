from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arg_func', [dict, Series])
def test_map_dict_ignore_na(arg_func):
    mapping = arg_func({1: 10, np.nan: 42})
    ser = Series([1, np.nan, 2])
    result = ser.map(mapping, na_action='ignore')
    expected = Series([10, np.nan, np.nan])
    tm.assert_series_equal(result, expected)