import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_categorical_transformers(request, transformation_func, observed, sort, as_index):
    if transformation_func == 'fillna':
        msg = 'GH#49651 fillna may incorrectly reorders results when dropna=False'
        request.applymarker(pytest.mark.xfail(reason=msg, strict=False))
    values = np.append(np.random.default_rng(2).choice([1, 2, None], size=19), None)
    df = pd.DataFrame({'x': pd.Categorical(values, categories=[1, 2, 3]), 'y': range(20)})
    args = get_groupby_method_args(transformation_func, df)
    null_group_values = df[df['x'].isnull()]['y']
    if transformation_func == 'cumcount':
        null_group_data = list(range(len(null_group_values)))
    elif transformation_func == 'ngroup':
        if sort:
            if observed:
                na_group = df['x'].nunique(dropna=False) - 1
            else:
                na_group = df['x'].nunique(dropna=False) - 1
        else:
            na_group = df.iloc[:null_group_values.index[0]]['x'].nunique()
        null_group_data = len(null_group_values) * [na_group]
    else:
        null_group_data = getattr(null_group_values, transformation_func)(*args)
    null_group_result = pd.DataFrame({'y': null_group_data})
    gb_keepna = df.groupby('x', dropna=False, observed=observed, sort=sort, as_index=as_index)
    gb_dropna = df.groupby('x', dropna=True, observed=observed, sort=sort)
    msg = "The default fill_method='ffill' in DataFrameGroupBy.pct_change is deprecated"
    if transformation_func == 'pct_change':
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = getattr(gb_keepna, 'pct_change')(*args)
    else:
        result = getattr(gb_keepna, transformation_func)(*args)
    expected = getattr(gb_dropna, transformation_func)(*args)
    for iloc, value in zip(df[df['x'].isnull()].index.tolist(), null_group_result.values.ravel()):
        if expected.ndim == 1:
            expected.iloc[iloc] = value
        else:
            expected.iloc[iloc, 0] = value
    if transformation_func == 'ngroup':
        expected[df['x'].notnull() & expected.ge(na_group)] += 1
    if transformation_func not in ('rank', 'diff', 'pct_change', 'shift'):
        expected = expected.astype('int64')
    tm.assert_equal(result, expected)