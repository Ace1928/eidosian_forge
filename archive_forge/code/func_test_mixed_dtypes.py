import ctypes
import math
import pytest
import pandas as pd
def test_mixed_dtypes(df_from_dict):
    df = df_from_dict({'a': [1, 2, 3], 'b': [3, 4, 5], 'c': [1.5, 2.5, 3.5], 'd': [9, 10, 11], 'e': [True, False, True], 'f': ['a', '', 'c']})
    dfX = df.__dataframe__()
    columns = {'a': 0, 'b': 0, 'c': 2, 'd': 0, 'e': 20, 'f': 21}
    for column, kind in columns.items():
        colX = dfX.get_column_by_name(column)
        assert colX.null_count == 0
        assert isinstance(colX.null_count, int)
        assert colX.size() == 3
        assert colX.offset == 0
        assert colX.dtype[0] == kind
    assert dfX.get_column_by_name('c').dtype[1] == 64