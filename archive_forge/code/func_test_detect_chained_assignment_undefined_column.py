from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_undefined_column(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame(np.arange(0, 9), columns=['count'])
    df['group'] = 'b'
    df_original = df.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df.iloc[0:5]['group'] = 'a'
        tm.assert_frame_equal(df, df_original)
    elif warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df.iloc[0:5]['group'] = 'a'
    else:
        with pytest.raises(SettingWithCopyError, match=msg):
            with tm.raises_chained_assignment_error():
                df.iloc[0:5]['group'] = 'a'