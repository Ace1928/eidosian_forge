from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_option', ['top', 'bottom'])
def test_groupby_op_with_nullables(na_option):
    df = DataFrame({'x': [None]}, dtype='Float64')
    result = df.groupby('x', dropna=False)['x'].rank(method='min', na_option=na_option)
    expected = Series([1.0], dtype='Float64', name=result.name)
    tm.assert_series_equal(result, expected)