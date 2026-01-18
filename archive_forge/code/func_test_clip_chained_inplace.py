import numpy as np
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_clip_chained_inplace(using_copy_on_write):
    df = DataFrame({'a': [1, 4, 2], 'b': 1})
    df_orig = df.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['a'].clip(1, 2, inplace=True)
        tm.assert_frame_equal(df, df_orig)
        with tm.raises_chained_assignment_error():
            df[['a']].clip(1, 2, inplace=True)
        tm.assert_frame_equal(df, df_orig)
    else:
        with tm.assert_produces_warning(FutureWarning, match='inplace method'):
            df['a'].clip(1, 2, inplace=True)
        with tm.assert_produces_warning(None):
            with option_context('mode.chained_assignment', None):
                df[['a']].clip(1, 2, inplace=True)
        with tm.assert_produces_warning(None):
            with option_context('mode.chained_assignment', None):
                df[df['a'] > 1].clip(1, 2, inplace=True)