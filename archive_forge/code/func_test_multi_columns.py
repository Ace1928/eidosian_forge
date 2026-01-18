import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('input_subset, expected_dict, expected_index', [(list('AC'), {'A': pd.Series([0, 1, 2, np.nan, np.nan, 3, 4, np.nan], index=list('aaabcdde'), dtype=object), 'B': 1, 'C': ['a', 'b', 'c', 'foo', np.nan, 'd', 'e', np.nan]}, list('aaabcdde')), (list('A'), {'A': pd.Series([0, 1, 2, np.nan, np.nan, 3, 4, np.nan], index=list('aaabcdde'), dtype=object), 'B': 1, 'C': [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c'], 'foo', [], ['d', 'e'], ['d', 'e'], np.nan]}, list('aaabcdde'))])
def test_multi_columns(input_subset, expected_dict, expected_index):
    df = pd.DataFrame({'A': [[0, 1, 2], np.nan, [], (3, 4), np.nan], 'B': 1, 'C': [['a', 'b', 'c'], 'foo', [], ['d', 'e'], np.nan]}, index=list('abcde'))
    result = df.explode(input_subset)
    expected = pd.DataFrame(expected_dict, expected_index)
    tm.assert_frame_equal(result, expected)