import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('df1,df2,msg', [(DataFrame.from_records({'a': [1, 2], 'c': ['l1', 'l2']}, index=['a']), DataFrame.from_records({'a': [1.0, 2.0], 'c': ['l1', 'l2']}, index=['a']), 'DataFrame\\.index are different'), (DataFrame.from_records({'a': [1, 2], 'b': [2.1, 1.5], 'c': ['l1', 'l2']}, index=['a', 'b']), DataFrame.from_records({'a': [1.0, 2.0], 'b': [2.1, 1.5], 'c': ['l1', 'l2']}, index=['a', 'b']), 'MultiIndex level \\[0\\] are different')])
def test_frame_equal_index_dtype_mismatch(df1, df2, msg, check_index_type):
    kwargs = {'check_index_type': check_index_type}
    if check_index_type:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, **kwargs)
    else:
        tm.assert_frame_equal(df1, df2, **kwargs)