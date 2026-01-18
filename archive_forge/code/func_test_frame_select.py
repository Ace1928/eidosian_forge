import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_frame_select(setup_path):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    with ensure_clean_store(setup_path) as store:
        store.put('frame', df, format='table')
        date = df.index[len(df) // 2]
        crit1 = Term('index>=date')
        assert crit1.env.scope['date'] == date
        crit2 = "columns=['A', 'D']"
        crit3 = 'columns=A'
        result = store.select('frame', [crit1, crit2])
        expected = df.loc[date:, ['A', 'D']]
        tm.assert_frame_equal(result, expected)
        result = store.select('frame', [crit3])
        expected = df.loc[:, ['A']]
        tm.assert_frame_equal(result, expected)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        store.append('df_time', df)
        msg = 'day is out of range for month: 0'
        with pytest.raises(ValueError, match=msg):
            store.select('df_time', 'index>0')