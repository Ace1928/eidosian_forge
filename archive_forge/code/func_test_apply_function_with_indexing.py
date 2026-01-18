import numpy as np
import pandas as pd
import pandas._testing as tm
def test_apply_function_with_indexing(warn_copy_on_write):
    df = pd.DataFrame({'col1': ['A', 'A', 'A', 'B', 'B', 'B'], 'col2': [1, 2, 3, 4, 5, 6]})

    def fn(x):
        x.loc[x.index[-1], 'col2'] = 0
        return x.col2
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg, raise_on_extra_warnings=not warn_copy_on_write):
        result = df.groupby(['col1'], as_index=False).apply(fn)
    expected = pd.Series([1, 2, 0, 4, 5, 0], index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (0, 2), (1, 3), (1, 4), (1, 5)]), name='col2')
    tm.assert_series_equal(result, expected)