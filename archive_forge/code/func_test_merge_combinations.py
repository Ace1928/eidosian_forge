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
@pytest.mark.parametrize('how', ['left', 'right', 'inner', 'outer'])
@pytest.mark.parametrize('sort', [True, False])
@pytest.mark.parametrize('on_index', [True, False])
@pytest.mark.parametrize('left_unique', [True, False])
@pytest.mark.parametrize('left_monotonic', [True, False])
@pytest.mark.parametrize('right_unique', [True, False])
@pytest.mark.parametrize('right_monotonic', [True, False])
def test_merge_combinations(how, sort, on_index, left_unique, left_monotonic, right_unique, right_monotonic):
    left = [2, 3]
    if left_unique:
        left.append(4 if left_monotonic else 1)
    else:
        left.append(3 if left_monotonic else 2)
    right = [2, 3]
    if right_unique:
        right.append(4 if right_monotonic else 1)
    else:
        right.append(3 if right_monotonic else 2)
    left = DataFrame({'key': left})
    right = DataFrame({'key': right})
    if on_index:
        left = left.set_index('key')
        right = right.set_index('key')
        on_kwargs = {'left_index': True, 'right_index': True}
    else:
        on_kwargs = {'on': 'key'}
    result = merge(left, right, how=how, sort=sort, **on_kwargs)
    if on_index:
        left = left.reset_index()
        right = right.reset_index()
    if how in ['left', 'right', 'inner']:
        if how in ['left', 'inner']:
            expected, other, other_unique = (left, right, right_unique)
        else:
            expected, other, other_unique = (right, left, left_unique)
        if how == 'inner':
            keep_values = set(left['key'].values).intersection(right['key'].values)
            keep_mask = expected['key'].isin(keep_values)
            expected = expected[keep_mask]
        if sort:
            expected = expected.sort_values('key')
        if not other_unique:
            other_value_counts = other['key'].value_counts()
            repeats = other_value_counts.reindex(expected['key'].values, fill_value=1)
            repeats = repeats.astype(np.intp)
            expected = expected['key'].repeat(repeats.values)
            expected = expected.to_frame()
    elif how == 'outer':
        left_counts = left['key'].value_counts()
        right_counts = right['key'].value_counts()
        expected_counts = left_counts.mul(right_counts, fill_value=1)
        expected_counts = expected_counts.astype(np.intp)
        expected = expected_counts.index.values.repeat(expected_counts.values)
        expected = DataFrame({'key': expected})
        expected = expected.sort_values('key')
    if on_index:
        expected = expected.set_index('key')
    else:
        expected = expected.reset_index(drop=True)
    tm.assert_frame_equal(result, expected)