import ctypes
import math
import pytest
import modin.pandas as pd
def test_na_float(df_from_dict):
    df = df_from_dict({'a': [1.0, math.nan, 2.0]})
    dfX = df.__dataframe__()
    colX = dfX.get_column_by_name('a')
    assert colX.null_count == 1