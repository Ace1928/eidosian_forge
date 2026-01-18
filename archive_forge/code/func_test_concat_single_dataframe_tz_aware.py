import pytest
import pandas.core.dtypes.concat as _concat
import pandas as pd
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('copy', [True, False])
def test_concat_single_dataframe_tz_aware(copy):
    df = pd.DataFrame({'timestamp': [pd.Timestamp('2020-04-08 09:00:00.709949+0000', tz='UTC')]})
    expected = df.copy()
    result = pd.concat([df], copy=copy)
    tm.assert_frame_equal(result, expected)