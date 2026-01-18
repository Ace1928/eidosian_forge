import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_escapable_characters(self):
    x = np.random.default_rng(2).standard_normal(3)
    df = DataFrame({'A(quarterly)1970': {0: 'a', 1: 'b', 2: 'c'}, 'A(quarterly)1980': {0: 'd', 1: 'e', 2: 'f'}, 'B(quarterly)1970': {0: 2.5, 1: 1.2, 2: 0.7}, 'B(quarterly)1980': {0: 3.2, 1: 1.3, 2: 0.1}, 'X': dict(zip(range(3), x))})
    df['id'] = df.index
    exp_data = {'X': x.tolist() + x.tolist(), 'A(quarterly)': ['a', 'b', 'c', 'd', 'e', 'f'], 'B(quarterly)': [2.5, 1.2, 0.7, 3.2, 1.3, 0.1], 'year': [1970, 1970, 1970, 1980, 1980, 1980], 'id': [0, 1, 2, 0, 1, 2]}
    expected = DataFrame(exp_data)
    expected = expected.set_index(['id', 'year'])[['X', 'A(quarterly)', 'B(quarterly)']]
    result = wide_to_long(df, ['A(quarterly)', 'B(quarterly)'], i='id', j='year')
    tm.assert_frame_equal(result, expected)