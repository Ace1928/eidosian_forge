import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_data_frame_value_counts_dropna_false(nulls_fixture):
    df = pd.DataFrame({'first_name': ['John', 'Anne', 'John', 'Beth'], 'middle_name': ['Smith', nulls_fixture, nulls_fixture, 'Louise']})
    result = df.value_counts(dropna=False)
    expected = pd.Series(data=[1, 1, 1, 1], index=pd.MultiIndex(levels=[pd.Index(['Anne', 'Beth', 'John']), pd.Index(['Louise', 'Smith', np.nan])], codes=[[0, 1, 2, 2], [2, 0, 1, 2]], names=['first_name', 'middle_name']), name='count')
    tm.assert_series_equal(result, expected)