from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('index_kind', ['range', 'single', 'multi'])
@pytest.mark.parametrize('method', ['head', 'tail'])
@pytest.mark.parametrize('ordered', [True, False])
def test_category_order_head_tail(as_index, sort, observed, method, index_kind, ordered):
    df = DataFrame({'a': Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered), 'b': range(4)})
    if index_kind == 'range':
        keys = ['a']
    elif index_kind == 'single':
        keys = ['a']
        df = df.set_index(keys)
    elif index_kind == 'multi':
        keys = ['a', 'a2']
        df['a2'] = df['a']
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result = getattr(gb, method)()
    if index_kind == 'range':
        result = op_result['a'].cat.categories
    else:
        result = op_result.index.get_level_values('a').categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)
    if index_kind == 'multi':
        result = op_result.index.get_level_values('a2').categories
        tm.assert_index_equal(result, expected)