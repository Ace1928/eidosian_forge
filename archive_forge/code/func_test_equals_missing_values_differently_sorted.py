import numpy as np
import pytest
from pandas.core.dtypes.common import is_any_real_numeric_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_equals_missing_values_differently_sorted():
    mi1 = MultiIndex.from_tuples([(81.0, np.nan), (np.nan, np.nan)])
    mi2 = MultiIndex.from_tuples([(np.nan, np.nan), (81.0, np.nan)])
    assert not mi1.equals(mi2)
    mi2 = MultiIndex.from_tuples([(81.0, np.nan), (np.nan, np.nan)])
    assert mi1.equals(mi2)