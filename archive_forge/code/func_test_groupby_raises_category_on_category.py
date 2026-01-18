import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_category_on_category(how, by, groupby_series, groupby_func, observed, using_copy_on_write, df_with_cat_col):
    df = df_with_cat_col
    df['a'] = Categorical(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c'], categories=['a', 'b', 'c', 'd'], ordered=True)
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by, observed=observed)
    if groupby_series:
        gb = gb['d']
        if groupby_func == 'corrwith':
            assert not hasattr(gb, 'corrwith')
            return
    empty_groups = not observed and any((group.empty for group in gb.groups.values()))
    if not observed and how != 'transform' and isinstance(by, list) and isinstance(by[0], str) and (by == ['a', 'b']):
        assert not empty_groups
        empty_groups = True
    if how == 'transform':
        empty_groups = False
    klass, msg = {'all': (None, ''), 'any': (None, ''), 'bfill': (None, ''), 'corrwith': (TypeError, "unsupported operand type\\(s\\) for \\*: 'Categorical' and 'int'"), 'count': (None, ''), 'cumcount': (None, ''), 'cummax': ((NotImplementedError, TypeError), '(cummax is not supported for category dtype|category dtype not supported|category type does not support cummax operations)'), 'cummin': ((NotImplementedError, TypeError), '(cummin is not supported for category dtype|category dtype not supported|category type does not support cummin operations)'), 'cumprod': ((NotImplementedError, TypeError), '(cumprod is not supported for category dtype|category dtype not supported|category type does not support cumprod operations)'), 'cumsum': ((NotImplementedError, TypeError), '(cumsum is not supported for category dtype|category dtype not supported|category type does not support cumsum operations)'), 'diff': (TypeError, 'unsupported operand type'), 'ffill': (None, ''), 'fillna': (TypeError, 'Cannot setitem on a Categorical with a new category \\(0\\), set the categories first') if not using_copy_on_write else (None, ''), 'first': (None, ''), 'idxmax': (ValueError, 'empty group due to unobserved categories') if empty_groups else (None, ''), 'idxmin': (ValueError, 'empty group due to unobserved categories') if empty_groups else (None, ''), 'last': (None, ''), 'max': (None, ''), 'mean': (TypeError, "category dtype does not support aggregation 'mean'"), 'median': (TypeError, "category dtype does not support aggregation 'median'"), 'min': (None, ''), 'ngroup': (None, ''), 'nunique': (None, ''), 'pct_change': (TypeError, 'unsupported operand type'), 'prod': (TypeError, 'category type does not support prod operations'), 'quantile': (TypeError, ''), 'rank': (None, ''), 'sem': (TypeError, '|'.join(["'Categorical' .* does not support reduction 'sem'", "category dtype does not support aggregation 'sem'"])), 'shift': (None, ''), 'size': (None, ''), 'skew': (TypeError, '|'.join(['category type does not support skew operations', "dtype category does not support reduction 'skew'"])), 'std': (TypeError, '|'.join(["'Categorical' .* does not support reduction 'std'", "category dtype does not support aggregation 'std'"])), 'sum': (TypeError, 'category type does not support sum operations'), 'var': (TypeError, '|'.join(["'Categorical' .* does not support reduction 'var'", "category dtype does not support aggregation 'var'"]))}[groupby_func]
    if groupby_func == 'fillna':
        kind = 'Series' if groupby_series else 'DataFrame'
        warn_msg = f'{kind}GroupBy.fillna is deprecated'
    else:
        warn_msg = ''
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg)