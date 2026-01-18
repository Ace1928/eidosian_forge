import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('left, right, on, left_by, right_by, expected', [(DataFrame({'G': ['g', 'g'], 'H': ['h', 'h'], 'T': [1, 3]}), DataFrame({'T': [2], 'E': [1]}), ['T'], ['G', 'H'], None, DataFrame({'G': ['g'] * 3, 'H': ['h'] * 3, 'T': [1, 2, 3], 'E': [np.nan, 1.0, np.nan]})), (DataFrame({'G': ['g', 'g'], 'H': ['h', 'h'], 'T': [1, 3]}), DataFrame({'T': [2], 'E': [1]}), 'T', ['G', 'H'], None, DataFrame({'G': ['g'] * 3, 'H': ['h'] * 3, 'T': [1, 2, 3], 'E': [np.nan, 1.0, np.nan]})), (DataFrame({'T': [2], 'E': [1]}), DataFrame({'G': ['g', 'g'], 'H': ['h', 'h'], 'T': [1, 3]}), ['T'], None, ['G', 'H'], DataFrame({'T': [1, 2, 3], 'E': [np.nan, 1.0, np.nan], 'G': ['g'] * 3, 'H': ['h'] * 3}))])
def test_list_type_by(self, left, right, on, left_by, right_by, expected):
    result = merge_ordered(left=left, right=right, on=on, left_by=left_by, right_by=right_by)
    tm.assert_frame_equal(result, expected)