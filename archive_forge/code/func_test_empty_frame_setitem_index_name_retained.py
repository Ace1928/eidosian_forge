import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_empty_frame_setitem_index_name_retained(self):
    df = DataFrame({}, index=pd.RangeIndex(0, name='df_index'))
    series = Series(1.23, index=pd.RangeIndex(4, name='series_index'))
    df['series'] = series
    expected = DataFrame({'series': [1.23] * 4}, index=pd.RangeIndex(4, name='df_index'), columns=Index(['series'], dtype=object))
    tm.assert_frame_equal(df, expected)