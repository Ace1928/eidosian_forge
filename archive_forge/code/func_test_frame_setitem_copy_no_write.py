import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_frame_setitem_copy_no_write(multiindex_dataframe_random_data, using_copy_on_write, warn_copy_on_write):
    frame = multiindex_dataframe_random_data.T
    expected = frame
    df = frame.copy()
    if using_copy_on_write or warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['foo']['one'] = 2
    else:
        msg = 'A value is trying to be set on a copy of a slice from a DataFrame'
        with pytest.raises(SettingWithCopyError, match=msg):
            with tm.raises_chained_assignment_error():
                df['foo']['one'] = 2
    result = df
    tm.assert_frame_equal(result, expected)