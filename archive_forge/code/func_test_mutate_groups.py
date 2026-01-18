import numpy as np
import pandas as pd
import pandas._testing as tm
def test_mutate_groups():
    df = pd.DataFrame({'cat1': ['a'] * 8 + ['b'] * 6, 'cat2': ['c'] * 2 + ['d'] * 2 + ['e'] * 2 + ['f'] * 2 + ['c'] * 2 + ['d'] * 2 + ['e'] * 2, 'cat3': [f'g{x}' for x in range(1, 15)], 'val': np.random.default_rng(2).integers(100, size=14)})

    def f_copy(x):
        x = x.copy()
        x['rank'] = x.val.rank(method='min')
        return x.groupby('cat2')['rank'].min()

    def f_no_copy(x):
        x['rank'] = x.val.rank(method='min')
        return x.groupby('cat2')['rank'].min()
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        grpby_copy = df.groupby('cat1').apply(f_copy)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        grpby_no_copy = df.groupby('cat1').apply(f_no_copy)
    tm.assert_series_equal(grpby_copy, grpby_no_copy)