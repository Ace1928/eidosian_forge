from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_category_string():
    a = Series(['a', 'b', 'c', 'd'])
    b = Series(['B', 'C', 'D', 'E'], dtype='category', index=pd.CategoricalIndex(['b', 'c', 'd', 'e']))
    c = Series(['B', 'C', 'D', 'E'], index=Index(['b', 'c', 'd', 'e']))
    exp = Series(pd.Categorical([np.nan, 'B', 'C', 'D'], categories=['B', 'C', 'D', 'E']))
    tm.assert_series_equal(a.map(b), exp)
    exp = Series([np.nan, 'B', 'C', 'D'])
    tm.assert_series_equal(a.map(c), exp)