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
@pytest.mark.parametrize('kwarg', [{'left_index': True, 'right_index': True}, {'left_index': True, 'right_on': 'x'}, {'left_on': 'a', 'right_index': True}, {'left_on': 'a', 'right_on': 'x'}])
def test_merge_left_empty_right_empty(self, join_type, kwarg):
    left = DataFrame(columns=['a', 'b', 'c'])
    right = DataFrame(columns=['x', 'y', 'z'])
    exp_in = DataFrame(columns=['a', 'b', 'c', 'x', 'y', 'z'], dtype=object)
    result = merge(left, right, how=join_type, **kwarg)
    tm.assert_frame_equal(result, exp_in)