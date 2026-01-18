import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('index_kind', ['range', 'single', 'multi'])
def test_categorical_reducers(reduction_func, observed, sort, as_index, index_kind):
    values = np.append(np.random.default_rng(2).choice([1, 2, None], size=19), None)
    df = pd.DataFrame({'x': pd.Categorical(values, categories=[1, 2, 3]), 'y': range(20)})
    df_filled = df.copy()
    df_filled['x'] = pd.Categorical(values, categories=[1, 2, 3, 4]).fillna(4)
    if index_kind == 'range':
        keys = ['x']
    elif index_kind == 'single':
        keys = ['x']
        df = df.set_index('x')
        df_filled = df_filled.set_index('x')
    else:
        keys = ['x', 'x2']
        df['x2'] = df['x']
        df = df.set_index(['x', 'x2'])
        df_filled['x2'] = df_filled['x']
        df_filled = df_filled.set_index(['x', 'x2'])
    args = get_groupby_method_args(reduction_func, df)
    args_filled = get_groupby_method_args(reduction_func, df_filled)
    if reduction_func == 'corrwith' and index_kind == 'range':
        args = (args[0].drop(columns=keys),)
        args_filled = (args_filled[0].drop(columns=keys),)
    gb_keepna = df.groupby(keys, dropna=False, observed=observed, sort=sort, as_index=as_index)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            getattr(gb_keepna, reduction_func)(*args)
        return
    gb_filled = df_filled.groupby(keys, observed=observed, sort=sort, as_index=True)
    expected = getattr(gb_filled, reduction_func)(*args_filled).reset_index()
    expected['x'] = expected['x'].cat.remove_categories([4])
    if index_kind == 'multi':
        expected['x2'] = expected['x2'].cat.remove_categories([4])
    if as_index:
        if index_kind == 'multi':
            expected = expected.set_index(['x', 'x2'])
        else:
            expected = expected.set_index('x')
    elif index_kind != 'range' and reduction_func != 'size':
        expected = expected.drop(columns='x')
        if index_kind == 'multi':
            expected = expected.drop(columns='x2')
    if reduction_func in ('idxmax', 'idxmin') and index_kind != 'range':
        values = expected['y'].values.tolist()
        if index_kind == 'single':
            values = [np.nan if e == 4 else e for e in values]
            expected['y'] = pd.Categorical(values, categories=[1, 2, 3])
        else:
            values = [(np.nan, np.nan) if e == (4, 4) else e for e in values]
            expected['y'] = values
    if reduction_func == 'size':
        expected = expected.rename(columns={0: 'size'})
        if as_index:
            expected = expected['size'].rename(None)
    if as_index or index_kind == 'range' or reduction_func == 'size':
        warn = None
    else:
        warn = FutureWarning
    msg = 'A grouping .* was excluded from the result'
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(gb_keepna, reduction_func)(*args)
    tm.assert_equal(result, expected)