from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_groupby_datetimeindex():
    data = [['A', 10], ['B', 20], ['B', 30], ['C', 40], ['C', 50]]
    df = DataFrame(data, columns=['Name', 'Value'], index=pd.date_range('2020-09-01', '2020-09-05'))
    result = df.groupby('Name').sum()
    expected = DataFrame({'Name': ['A', 'B', 'C'], 'Value': [10, 50, 90]})
    expected.set_index('Name', inplace=True)
    tm.assert_frame_equal(result, expected)