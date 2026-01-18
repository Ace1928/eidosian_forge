import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_compare_datetime64_and_string():
    data = [{'a': '2015-07-01', 'b': '08335394550'}, {'a': '2015-07-02', 'b': '+49 (0) 0345 300033'}, {'a': '2015-07-03', 'b': '+49(0)2598 04457'}, {'a': '2015-07-04', 'b': '0741470003'}, {'a': '2015-07-05', 'b': '04181 83668'}]
    dtypes = {'a': 'datetime64[ns]', 'b': 'string'}
    df = pd.DataFrame(data=data).astype(dtypes)
    result_eq1 = df['a'].eq(df['b'])
    result_eq2 = df['a'] == df['b']
    result_neq = df['a'] != df['b']
    expected_eq = pd.Series([False] * 5)
    expected_neq = pd.Series([True] * 5)
    tm.assert_series_equal(result_eq1, expected_eq)
    tm.assert_series_equal(result_eq2, expected_eq)
    tm.assert_series_equal(result_neq, expected_neq)