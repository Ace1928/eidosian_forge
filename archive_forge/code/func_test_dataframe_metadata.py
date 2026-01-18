import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dataframe_metadata(self):
    df = tm.SubclassedDataFrame({'X': [1, 2, 3], 'Y': [1, 2, 3]}, index=['a', 'b', 'c'])
    df.testattr = 'XXX'
    assert df.testattr == 'XXX'
    assert df[['X']].testattr == 'XXX'
    assert df.loc[['a', 'b'], :].testattr == 'XXX'
    assert df.iloc[[0, 1], :].testattr == 'XXX'
    assert df.iloc[0:1, :].testattr == 'XXX'
    unpickled = tm.round_trip_pickle(df)
    tm.assert_frame_equal(df, unpickled)
    assert df._metadata == unpickled._metadata
    assert df.testattr == unpickled.testattr