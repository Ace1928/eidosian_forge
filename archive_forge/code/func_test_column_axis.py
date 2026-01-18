import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_column_axis(column_group_df):
    msg = 'DataFrame.groupby with axis=1'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        g = column_group_df.groupby(column_group_df.iloc[1], axis=1)
    result = g._positional_selector[1:-1]
    expected = column_group_df.iloc[:, [1, 3]]
    tm.assert_frame_equal(result, expected)