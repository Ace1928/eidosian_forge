from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_multiindex_dfs_with_deepcopy(self):
    example_multiindex1 = MultiIndex.from_product([['a'], ['b']])
    example_dataframe1 = DataFrame([0], index=example_multiindex1)
    example_multiindex2 = MultiIndex.from_product([['a'], ['c']])
    example_dataframe2 = DataFrame([1], index=example_multiindex2)
    example_dict = {'s1': example_dataframe1, 's2': example_dataframe2}
    expected_index = MultiIndex(levels=[['s1', 's2'], ['a'], ['b', 'c']], codes=[[0, 1], [0, 0], [0, 1]], names=['testname', None, None])
    expected = DataFrame([[0], [1]], index=expected_index)
    result_copy = concat(deepcopy(example_dict), names=['testname'])
    tm.assert_frame_equal(result_copy, expected)
    result_no_copy = concat(example_dict, names=['testname'])
    tm.assert_frame_equal(result_no_copy, expected)