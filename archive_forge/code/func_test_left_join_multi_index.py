import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('sort', [True, False])
def test_left_join_multi_index(self, sort, infer_string):
    with option_context('future.infer_string', infer_string):
        icols = ['1st', '2nd', '3rd']

        def bind_cols(df):
            iord = lambda a: 0 if a != a else ord(a)
            f = lambda ts: ts.map(iord) - ord('a')
            return f(df['1st']) + f(df['3rd']) * 100.0 + df['2nd'].fillna(0) * 10

        def run_asserts(left, right, sort):
            res = left.join(right, on=icols, how='left', sort=sort)
            assert len(left) < len(res) + 1
            assert not res['4th'].isna().any()
            assert not res['5th'].isna().any()
            tm.assert_series_equal(res['4th'], -res['5th'], check_names=False)
            result = bind_cols(res.iloc[:, :-2])
            tm.assert_series_equal(res['4th'], result, check_names=False)
            assert result.name is None
            if sort:
                tm.assert_frame_equal(res, res.sort_values(icols, kind='mergesort'))
            out = merge(left, right.reset_index(), on=icols, sort=sort, how='left')
            res.index = RangeIndex(len(res))
            tm.assert_frame_equal(out, res)
        lc = list(map(chr, np.arange(ord('a'), ord('z') + 1)))
        left = DataFrame(np.random.default_rng(2).choice(lc, (50, 2)), columns=['1st', '3rd'])
        left.insert(1, '2nd', np.random.default_rng(2).integers(0, 10, len(left)).astype('float'))
        i = np.random.default_rng(2).permutation(len(left))
        right = left.iloc[i].copy()
        left['4th'] = bind_cols(left)
        right['5th'] = -bind_cols(right)
        right.set_index(icols, inplace=True)
        run_asserts(left, right, sort)
        left.loc[1::4, '1st'] = np.nan
        left.loc[2::5, '2nd'] = np.nan
        left.loc[3::6, '3rd'] = np.nan
        left['4th'] = bind_cols(left)
        i = np.random.default_rng(2).permutation(len(left))
        right = left.iloc[i, :-1]
        right['5th'] = -bind_cols(right)
        right.set_index(icols, inplace=True)
        run_asserts(left, right, sort)