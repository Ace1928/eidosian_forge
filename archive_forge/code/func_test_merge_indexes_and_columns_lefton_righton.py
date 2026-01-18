import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('left_on,right_on,how', [(['outer'], ['outer'], 'inner'), (['inner'], ['inner'], 'right'), (['outer', 'inner'], ['outer', 'inner'], 'left'), (['inner', 'outer'], ['inner', 'outer'], 'outer')])
def test_merge_indexes_and_columns_lefton_righton(left_df, right_df, left_on, right_on, how):
    expected = compute_expected(left_df, right_df, left_on=left_on, right_on=right_on, how=how)
    result = left_df.merge(right_df, left_on=left_on, right_on=right_on, how=how)
    tm.assert_frame_equal(result, expected, check_like=True)