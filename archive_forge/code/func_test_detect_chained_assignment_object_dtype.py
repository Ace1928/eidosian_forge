from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_object_dtype(self, using_array_manager, using_copy_on_write, warn_copy_on_write):
    expected = DataFrame({'A': [111, 'bbb', 'ccc'], 'B': [1, 2, 3]})
    df = DataFrame({'A': Series(['aaa', 'bbb', 'ccc'], dtype=object), 'B': [1, 2, 3]})
    df_original = df.copy()
    if not using_copy_on_write and (not warn_copy_on_write):
        with pytest.raises(SettingWithCopyError, match=msg):
            df.loc[0]['A'] = 111
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['A'][0] = 111
        tm.assert_frame_equal(df, df_original)
    elif warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['A'][0] = 111
        tm.assert_frame_equal(df, expected)
    elif not using_array_manager:
        with pytest.raises(SettingWithCopyError, match=msg):
            with tm.raises_chained_assignment_error():
                df['A'][0] = 111
        df.loc[0, 'A'] = 111
        tm.assert_frame_equal(df, expected)
    else:
        df['A'][0] = 111
        tm.assert_frame_equal(df, expected)