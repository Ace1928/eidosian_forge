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
def test_merge_multiindex_columns():
    letters = ['a', 'b', 'c', 'd']
    numbers = ['1', '2', '3']
    index = MultiIndex.from_product((letters, numbers), names=['outer', 'inner'])
    frame_x = DataFrame(columns=index)
    frame_x['id'] = ''
    frame_y = DataFrame(columns=index)
    frame_y['id'] = ''
    l_suf = '_x'
    r_suf = '_y'
    result = frame_x.merge(frame_y, on='id', suffixes=(l_suf, r_suf))
    tuples = [(letter + l_suf, num) for letter in letters for num in numbers]
    tuples += [('id', '')]
    tuples += [(letter + r_suf, num) for letter in letters for num in numbers]
    expected_index = MultiIndex.from_tuples(tuples, names=['outer', 'inner'])
    expected = DataFrame(columns=expected_index)
    tm.assert_frame_equal(result, expected, check_dtype=False)