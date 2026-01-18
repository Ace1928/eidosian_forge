from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_at_getitem_dt64tz_values(self):
    df = DataFrame({'name': ['John', 'Anderson'], 'date': [Timestamp(2017, 3, 13, 13, 32, 56), Timestamp(2017, 2, 16, 12, 10, 3)]})
    df['date'] = df['date'].dt.tz_localize('Asia/Shanghai')
    expected = Timestamp('2017-03-13 13:32:56+0800', tz='Asia/Shanghai')
    result = df.loc[0, 'date']
    assert result == expected
    result = df.at[0, 'date']
    assert result == expected