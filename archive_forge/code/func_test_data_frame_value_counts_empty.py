import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_data_frame_value_counts_empty():
    df_no_cols = pd.DataFrame()
    result = df_no_cols.value_counts()
    expected = pd.Series([], dtype=np.int64, name='count', index=np.array([], dtype=np.intp))
    tm.assert_series_equal(result, expected)