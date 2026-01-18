from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_left_empty_right_notempty(self):
    left = DataFrame(columns=['a', 'b', 'c'])
    right = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['x', 'y', 'z'])
    exp_out = DataFrame({'a': np.array([np.nan] * 3, dtype=object), 'b': np.array([np.nan] * 3, dtype=object), 'c': np.array([np.nan] * 3, dtype=object), 'x': [1, 4, 7], 'y': [2, 5, 8], 'z': [3, 6, 9]}, columns=['a', 'b', 'c', 'x', 'y', 'z'])
    exp_in = exp_out[0:0]

    def check1(exp, kwarg):
        result = merge(left, right, how='inner', **kwarg)
        tm.assert_frame_equal(result, exp)
        result = merge(left, right, how='left', **kwarg)
        tm.assert_frame_equal(result, exp)

    def check2(exp, kwarg):
        result = merge(left, right, how='right', **kwarg)
        tm.assert_frame_equal(result, exp)
        result = merge(left, right, how='outer', **kwarg)
        tm.assert_frame_equal(result, exp)
    for kwarg in [{'left_index': True, 'right_index': True}, {'left_index': True, 'right_on': 'x'}]:
        check1(exp_in, kwarg)
        check2(exp_out, kwarg)
    kwarg = {'left_on': 'a', 'right_index': True}
    check1(exp_in, kwarg)
    exp_out['a'] = [0, 1, 2]
    check2(exp_out, kwarg)
    kwarg = {'left_on': 'a', 'right_on': 'x'}
    check1(exp_in, kwarg)
    exp_out['a'] = np.array([np.nan] * 3, dtype=object)
    check2(exp_out, kwarg)