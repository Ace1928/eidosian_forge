import numpy as np
import pytest
from pandas.core.dtypes.common import is_any_real_numeric_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_equals_missing_values():
    i = MultiIndex.from_tuples([(0, pd.NaT), (0, pd.Timestamp('20130101'))])
    result = i[0:1].equals(i[0])
    assert not result
    result = i[1:2].equals(i[1])
    assert not result