import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_fillna_series_method(self, data_missing, fillna_method):
    fill_value = data_missing[1]
    if fillna_method == 'ffill':
        data_missing = data_missing[::-1]
    result = getattr(pd.Series(data_missing), fillna_method)()
    expected = pd.Series(data_missing._from_sequence([fill_value, fill_value], dtype=data_missing.dtype))
    tm.assert_series_equal(result, expected)