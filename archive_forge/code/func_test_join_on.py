import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
def test_join_on(self, target_source, infer_string):
    target, source = target_source
    merged = target.join(source, on='C')
    tm.assert_series_equal(merged['MergedA'], target['A'], check_names=False)
    tm.assert_series_equal(merged['MergedD'], target['D'], check_names=False)
    df = DataFrame({'key': ['a', 'a', 'b', 'b', 'c']})
    df2 = DataFrame({'value': [0, 1, 2]}, index=['a', 'b', 'c'])
    joined = df.join(df2, on='key')
    expected = DataFrame({'key': ['a', 'a', 'b', 'b', 'c'], 'value': [0, 0, 1, 1, 2]})
    tm.assert_frame_equal(joined, expected)
    df_a = DataFrame([[1], [2], [3]], index=['a', 'b', 'c'], columns=['one'])
    df_b = DataFrame([['foo'], ['bar']], index=[1, 2], columns=['two'])
    df_c = DataFrame([[1], [2]], index=[1, 2], columns=['three'])
    joined = df_a.join(df_b, on='one')
    joined = joined.join(df_c, on='one')
    assert np.isnan(joined['two']['c'])
    assert np.isnan(joined['three']['c'])
    with pytest.raises(KeyError, match="^'E'$"):
        target.join(source, on='E')
    source_copy = source.copy()
    msg = "You are trying to merge on float64 and object|string columns for key 'A'. If you wish to proceed you should use pd.concat"
    with pytest.raises(ValueError, match=msg):
        target.join(source_copy, on='A')