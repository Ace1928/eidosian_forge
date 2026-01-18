import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('by, expected_data', [[['id'], {'num': [100.0, 150.0, 150.0, 200.0]}], [['id', 'index'], {'date': [Timestamp('2018-01-01'), Timestamp('2018-01-02'), Timestamp('2018-01-01'), Timestamp('2018-01-02')], 'num': [100.0, 200.0, 150.0, 250.0]}]])
def test_as_index_false(self, by, expected_data, unit):
    data = [['A', '2018-01-01', 100.0], ['A', '2018-01-02', 200.0], ['B', '2018-01-01', 150.0], ['B', '2018-01-02', 250.0]]
    df = DataFrame(data, columns=['id', 'date', 'num'])
    df['date'] = df['date'].astype(f'M8[{unit}]')
    df = df.set_index(['date'])
    gp_by = [getattr(df, attr) for attr in by]
    result = df.groupby(gp_by, as_index=False).rolling(window=2, min_periods=1).mean()
    expected = {'id': ['A', 'A', 'B', 'B']}
    expected.update(expected_data)
    expected = DataFrame(expected, index=df.index)
    if 'date' in expected_data:
        expected['date'] = expected['date'].astype(f'M8[{unit}]')
    tm.assert_frame_equal(result, expected)