import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_data_frame_value_counts_empty_normalize():
    df_no_cols = pd.DataFrame()
    result = df_no_cols.value_counts(normalize=True)
    expected = pd.Series([], dtype=np.float64, name='proportion', index=np.array([], dtype=np.intp))
    tm.assert_series_equal(result, expected)