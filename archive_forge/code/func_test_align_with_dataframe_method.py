from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['backfill', 'bfill', 'pad', 'ffill', None])
def test_align_with_dataframe_method(method):
    ser = Series(range(3), index=range(3))
    df = pd.DataFrame(0.0, index=range(3), columns=range(3))
    msg = "The 'method', 'limit', and 'fill_axis' keywords in Series.align are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result_ser, result_df = ser.align(df, method=method)
    tm.assert_series_equal(result_ser, ser)
    tm.assert_frame_equal(result_df, df)