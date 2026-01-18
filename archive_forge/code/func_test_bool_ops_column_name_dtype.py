import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.mark.parametrize('test_input,expected', [(DataFrame([[0, 1, 2, 'aa'], [0, 1, 2, 'aa']], columns=['a', 'b', 'c', 'dtype']), DataFrame([[False, False], [False, False]], columns=['a', 'dtype'])), (DataFrame([[0, 3, 2, 'aa'], [0, 4, 2, 'aa'], [0, 1, 1, 'bb']], columns=['a', 'b', 'c', 'dtype']), DataFrame([[False, False], [False, False], [False, False]], columns=['a', 'dtype']))])
def test_bool_ops_column_name_dtype(self, test_input, expected):
    result = test_input.loc[:, ['a', 'dtype']].ne(test_input.loc[:, ['a', 'dtype']])
    tm.assert_frame_equal(result, expected)