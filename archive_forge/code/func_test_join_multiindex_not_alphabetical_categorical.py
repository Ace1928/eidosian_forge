import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('categories, values', [(['Y', 'X'], ['Y', 'X', 'X']), ([2, 1], [2, 1, 1]), ([2.5, 1.5], [2.5, 1.5, 1.5]), ([Timestamp('2020-12-31'), Timestamp('2019-12-31')], [Timestamp('2020-12-31'), Timestamp('2019-12-31'), Timestamp('2019-12-31')])])
def test_join_multiindex_not_alphabetical_categorical(categories, values):
    left = DataFrame({'first': ['A', 'A'], 'second': Categorical(categories, categories=categories), 'value': [1, 2]}).set_index(['first', 'second'])
    right = DataFrame({'first': ['A', 'A', 'B'], 'second': Categorical(values, categories=categories), 'value': [3, 4, 5]}).set_index(['first', 'second'])
    result = left.join(right, lsuffix='_left', rsuffix='_right')
    expected = DataFrame({'first': ['A', 'A'], 'second': Categorical(categories, categories=categories), 'value_left': [1, 2], 'value_right': [3, 4]}).set_index(['first', 'second'])
    tm.assert_frame_equal(result, expected)