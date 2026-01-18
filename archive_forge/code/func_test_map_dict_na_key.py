from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_dict_na_key():
    s = Series([1, 2, np.nan])
    expected = Series(['a', 'b', 'c'])
    result = s.map({1: 'a', 2: 'b', np.nan: 'c'})
    tm.assert_series_equal(result, expected)