from datetime import datetime
import numpy as np
from pandas import (
import pandas._testing as tm
def test_multiindex_datetime_columns():
    mi = MultiIndex.from_tuples([(to_datetime('02/29/2020'), to_datetime('03/01/2020'))], names=['a', 'b'])
    df = DataFrame([], columns=mi)
    expected_df = DataFrame([], columns=MultiIndex.from_arrays([[to_datetime('02/29/2020')], [to_datetime('03/01/2020')]], names=['a', 'b']))
    tm.assert_frame_equal(df, expected_df)