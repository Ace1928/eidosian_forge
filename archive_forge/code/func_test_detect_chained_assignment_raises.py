from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_raises(self, using_array_manager, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'A': Series(range(2), dtype='int64'), 'B': np.array(np.arange(2, 4), dtype=np.float64)})
    df_original = df.copy()
    assert df._is_copy is None
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['A'][0] = -5
        with tm.raises_chained_assignment_error():
            df['A'][1] = -6
        tm.assert_frame_equal(df, df_original)
    elif warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['A'][0] = -5
        with tm.raises_chained_assignment_error():
            df['A'][1] = np.nan
    elif not using_array_manager:
        with pytest.raises(SettingWithCopyError, match=msg):
            with tm.raises_chained_assignment_error():
                df['A'][0] = -5
        with pytest.raises(SettingWithCopyError, match=msg):
            with tm.raises_chained_assignment_error():
                df['A'][1] = np.nan
        assert df['A']._is_copy is None
    else:
        df['A'][0] = -5
        df['A'][1] = -6
        expected = DataFrame([[-5, 2], [-6, 3]], columns=list('AB'))
        expected['B'] = expected['B'].astype('float64')
        tm.assert_frame_equal(df, expected)