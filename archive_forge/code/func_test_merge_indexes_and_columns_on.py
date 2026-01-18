import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('on,how', [(['outer'], 'inner'), (['inner'], 'left'), (['outer', 'inner'], 'right'), (['inner', 'outer'], 'outer')])
def test_merge_indexes_and_columns_on(left_df, right_df, on, how):
    expected = compute_expected(left_df, right_df, on=on, how=how)
    result = left_df.merge(right_df, on=on, how=how)
    tm.assert_frame_equal(result, expected, check_like=True)